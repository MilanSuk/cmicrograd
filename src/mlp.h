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

typedef struct Neuron_s
{
	int num_inputs;
	Value **w;
	Value *b;
} Neuron;

void Neuron_init(Neuron *self, const int num_inputs, ValueAllocator *allocator)
{
	self->num_inputs = num_inputs;
	self->w = malloc(self->num_inputs * sizeof(Value *));
	for (int i = 0; i < self->num_inputs; i++)
		self->w[i] = VA_const(allocator, Std_random11());
	self->b = VA_const(allocator, Std_random11());
}

void Neuron_free(Neuron *self)
{
	memset(self->w, 0, self->num_inputs * sizeof(Value *));
	free(self->w);
}

Value *Neuron_build(Neuron *self, Value **x, ValueAllocator *allocator)
{
	// w * x + b
	Value *act = self->b;
	for (int i = 0; i < self->num_inputs; i++)
		act = VA_add(allocator, act, VA_mul(allocator, self->w[i], x[i]));

	act = VA_tanh(allocator, act);
	return act;
}

typedef struct Layer_s
{
	int num;
	Neuron *neurons;
	Value **outputs;
} Layer;

void Layer_init(Layer *self, const int num_inputs, const int num_outputs, ValueAllocator *allocator)
{
	self->num = num_outputs;
	self->neurons = malloc(self->num * sizeof(Neuron));
	self->outputs = malloc(self->num * sizeof(Value *));
	for (int i = 0; i < self->num; i++)
	{
		Neuron_init(&self->neurons[i], num_inputs, allocator);
		self->outputs[i] = 0;
	}
}

void Layer_free(Layer *self)
{
	for (int i = 0; i < self->num; i++)
		Neuron_free(&self->neurons[i]);
	memset(self->neurons, 0, self->num * sizeof(Neuron));
	free(self->neurons);

	memset(self->outputs, 0, self->num * sizeof(Value *));
	free(self->outputs);
}

Value **Layer_build(Layer *self, Value **x, ValueAllocator *allocator)
{
	for (int i = 0; i < self->num; i++)
		self->outputs[i] = Neuron_build(&self->neurons[i], x, allocator);
	return self->outputs;
}

typedef struct MLP_s
{
	int num_layers;
	Layer *layers;
} MLP;

MLP *MLP_new(const int num_inputs, const int *outputs, const int num_outputs, ValueAllocator *allocator)
{
	MLP *self = malloc(sizeof(MLP));

	self->num_layers = num_outputs;
	self->layers = malloc(self->num_layers * sizeof(Layer));

	if (num_outputs)
	{
		Layer_init(&self->layers[0], num_inputs, outputs[0], allocator); // 1st
		for (int i = 1; i < self->num_layers; i++)						 // starts with 1
			Layer_init(&self->layers[i], outputs[i - 1], outputs[i], allocator);
	}

	return self;
}

void MLP_delete(MLP *self)
{
	for (int i = 0; i < self->num_layers; i++)
		Layer_free(&self->layers[i]);
	memset(self->layers, 0, self->num_layers * sizeof(Layer));
	free(self->layers);

	memset(self, 0, sizeof(MLP));
	free(self);
}

Value **MLP_build(MLP *self, Value **x, ValueAllocator *allocator)
{
	for (int i = 0; i < self->num_layers; i++)
		x = Layer_build(&self->layers[i], x, allocator); // output 'x' is use as input 'x' to another layer
	return x;
}
