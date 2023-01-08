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

typedef struct TopoLayer_s
{
	Value **values;
	int num_values;
} TopoLayer;

typedef struct Topo_s
{
	TopoLayer *layers;
	int num_layers;
} Topo;

void _Topo_build(Topo *self, Value *v)
{
	if (v && !v->visited)
	{
		v->visited = 1;
		_Topo_build(self, v->prevs[0]);
		_Topo_build(self, v->prevs[1]);

		// add layer
		if (v->layer >= self->num_layers)
		{
			int old_num_layers = self->num_layers;
			self->num_layers = v->layer + 1;
			self->layers = realloc(self->layers, self->num_layers * sizeof(TopoLayer));
			memset(&self->layers[old_num_layers].values, 0, (self->num_layers - old_num_layers) * sizeof(TopoLayer));
		}
		// add value into layer
		{
			TopoLayer *layer = &self->layers[v->layer];
			layer->num_values++;
			layer->values = realloc(layer->values, layer->num_values * sizeof(Value *));
			layer->values[layer->num_values - 1] = v;
		}
	}
}

Topo *Topo_new(Value *result)
{
	Topo *self = malloc(sizeof(Topo));
	self->layers = 0;
	self->num_layers = 0;

	_Value_resetVisited(result);
	_Value_updateDepth(result);
	_Topo_build(self, result);

	return self;
}

void Topo_delete(Topo *self)
{
	for (int i = 0; i < self->num_layers; i++)
	{
		TopoLayer *layer = &self->layers[i];
		memset(layer->values, 0, layer->num_values * sizeof(Value *));
		free(layer->values);
	}

	memset(self->layers, 0, self->num_layers * sizeof(TopoLayer));
	free(self->layers);

	memset(self, 0, sizeof(Topo));
	free(self);
}

int Topo_numParameters(Topo *self)
{
	int n = 0;
	for (int i = 0; i < self->num_layers; i++)
		n += self->layers[i].num_values;
	return n;
}

void Topo_resetGrads(Topo *self)
{
	// zero
	for (int i = 0; i < self->num_layers; i++)
	{
		TopoLayer *layer = &self->layers[i];
		for (int i = 0; i < layer->num_values; i++)
			layer->values[i]->grad = 0;
	}

	// one
	if (self->num_layers)
	{
		TopoLayer *layer = &self->layers[self->num_layers - 1];
		for (int i = 0; i < layer->num_values; i++)
			layer->values[i]->grad = 1;
	}
}

void Topo_run(Topo *self)
{
	if (self->num_layers == 0)
		return;

	// forward
	for (int i = 0; i < self->num_layers; i++)
	{
		TopoLayer *layer = &self->layers[i];
		for (int ii = 0; ii < layer->num_values; ii++)
			Value_forward(layer->values[ii]);
	}

	// reset grads
	Topo_resetGrads(self);

	// backward
	for (int i = self->num_layers - 1; i >= 0; i--)
	{
		TopoLayer *layer = &self->layers[i];
		for (int ii = 0; ii < layer->num_values; ii++)
			Value_backward(layer->values[ii]);
	}
}

void Topo_update(Topo *self, const double val)
{
	for (int i = 0; i < self->num_layers; i++)
	{
		TopoLayer *layer = &self->layers[i];
		for (int ii = 0; ii < layer->num_values; ii++)
			layer->values[ii]->data += val * layer->values[ii]->grad;
	}
}

void Topo_print(Topo *self)
{
	printf("Num Layers: %d, Num Parameters: %d\n", self->num_layers, Topo_numParameters(self));
	for (int i = 0; i < self->num_layers; i++)
	{
		TopoLayer *layer = &self->layers[i];
		printf("[layer %d] Num parameters: %d\n", i, layer->num_values);
	}
}
