/*
Copyright 2022 Milan Suk

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

typedef struct TopoMT_s TopoMT;
typedef struct TopoThread_s
{
	TopoMT *parent;
	int i_thread;

	StdThread thread;

	OsSemaphore semaphore_new_work;
	OsSemaphore semaphore_work_done;

	volatile int forward_layer;
	volatile int backward_layer;
} TopoThread;

typedef struct TopoMT_s
{
	Topo *topo;

	TopoThread **threads;
	int num_threads;
} TopoMT;

StdThread_FUNC(TopoThread_loop, arg);
TopoThread *TopoThread_new(TopoMT *parent, const int i_thread)
{
	TopoThread *self = malloc(sizeof(TopoThread));
	self->parent = parent;
	self->i_thread = i_thread;

	OsSemaphore_init(&self->semaphore_new_work);
	OsSemaphore_init(&self->semaphore_work_done);

	StdThread_init(&self->thread, "TopoThread", &TopoThread_loop, self);

	return self;
}
void TopoThread_delete(TopoThread *self)
{
	self->thread.run = 0;
	OsSemaphore_trigger(&self->semaphore_new_work);
	StdThread_free(&self->thread);

	OsSemaphore_free(&self->semaphore_new_work);
	OsSemaphore_free(&self->semaphore_work_done);

	memset(self, 0, sizeof(TopoThread));
	free(self);
}

StdThread_FUNC(TopoThread_loop, arg)
{
	TopoThread *self = arg;
	while (self->thread.run)
	{
		const int NTHREADS = self->parent->num_threads;

		if (OsSemaphore_wait(&self->semaphore_new_work)) // waits for fork
		{
			if (self->forward_layer >= 0)
			{
				TopoLayer *layer = &self->parent->topo->layers[self->forward_layer];
				const int step = layer->num_values / NTHREADS + 1;
				const int st = step * self->i_thread;
				const int en = Std_bmin(layer->num_values, step * (self->i_thread + 1));
				for (int i = st; i < en; i++)
					Value_forward(layer->values[i]);

				self->forward_layer = -1;
				OsSemaphore_trigger(&self->semaphore_work_done); // work is done
			}
			else if (self->backward_layer >= 0)
			{
				TopoLayer *layer = &self->parent->topo->layers[self->backward_layer];
				const int step = layer->num_values / NTHREADS + 1;
				const int st = step * self->i_thread;
				const int en = Std_bmin(layer->num_values, step * (self->i_thread + 1));
				for (int i = st; i < en; i++)
					Value_backward(layer->values[i]);

				self->backward_layer = -1;
				OsSemaphore_trigger(&self->semaphore_work_done); // work is done
			}
		}
	}
	return 0;
}

TopoMT *TopoMT_new(int num_threads)
{
	TopoMT *self = malloc(sizeof(TopoMT));
	self->num_threads = (num_threads <= 0) ? Std_numberOfThreads() : num_threads;

	self->threads = malloc(self->num_threads * sizeof(TopoThread));
	for (int i = 0; i < self->num_threads; i++)
		self->threads[i] = TopoThread_new(self, i);
	return self;
}

void TopoMT_delete(TopoMT *self)
{
	for (int i = 0; i < self->num_threads; i++)
		TopoThread_delete(self->threads[i]);
	memset(self->threads, 0, self->num_threads * sizeof(TopoThread));
	free(self->threads);

	memset(self, 0, sizeof(TopoMT));
	free(self);
}

void TopoMT_run(TopoMT *self, Topo *topo)
{
	if (topo->num_layers == 0)
		return;

	self->topo = topo;

	// forward
	for (int i = 0; i < topo->num_layers; i++)
	{
		// sends work
		for (int t = 0; t < self->num_threads; t++)
		{
			self->threads[t]->forward_layer = i;
			OsSemaphore_trigger(&self->threads[t]->semaphore_new_work);
		}

		// waits until it's done
		for (int t = 0; t < self->num_threads; t++)
			OsSemaphore_wait(&self->threads[t]->semaphore_work_done);
	}

	// resets grads
	Topo_resetGrads(topo);

	// backward
	for (int i = topo->num_layers - 1; i >= 0; i--)
	{
		// sends work
		for (int t = 0; t < self->num_threads; t++)
		{
			self->threads[t]->backward_layer = i;
			OsSemaphore_trigger(&self->threads[t]->semaphore_new_work);
		}

		// waits until it's done
		for (int t = 0; t < self->num_threads; t++)
			OsSemaphore_wait(&self->threads[t]->semaphore_work_done);
	}
}
