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

typedef enum
{
	Value_OP_EMPTY,
	Value_OP_ADD,
	Value_OP_SUB,
	Value_OP_MUL,
	Value_OP_DIV,
	Value_OP_POW_CONST,
	Value_OP_NEG,
	Value_OP_TANH,
	Value_OP_RELU,
} Value_OP;

typedef struct Value_s
{
	double data;
	double grad;

	struct Value_s *prevs[2];

	unsigned char op : 7, visited : 1;
	unsigned int layer; // TODO: too much space - get rid of it
} Value;

Value *_Value_init(Value *self, const double data, const Value_OP op)
{
	self->data = data;
	self->grad = 0;

	self->prevs[0] = self->prevs[1] = 0;

	self->op = op;
	self->visited = 0;
	return self;
}

void Value_setPre(Value *self, const int i, Value *pre)
{
	self->prevs[i] = pre;
}
void Value_setPre2(Value *self, Value *a, Value *b)
{
	Value_setPre(self, 0, a);
	Value_setPre(self, 1, b);
}

void _Value_resetVisited(Value *v)
{
	if (v)
	{
		_Value_resetVisited(v->prevs[0]);
		_Value_resetVisited(v->prevs[1]);
		v->visited = 0;
		v->layer = 1000000000;
	}
}
int _Value_updateDepth(Value *v)
{
	int depth = 0;
	if (v)
	{
		depth = Std_bmax(_Value_updateDepth(v->prevs[0]), _Value_updateDepth(v->prevs[1]));
		v->layer = Std_bmin(v->layer, depth);
		depth++;
	}

	return depth;
}

void Value_forward(Value *self)
{
	switch (self->op)
	{
	case Value_OP_EMPTY:
		break;
	case Value_OP_ADD:
		self->data = self->prevs[0]->data + self->prevs[1]->data;
		break;
	case Value_OP_SUB:
		self->data = self->prevs[0]->data - self->prevs[1]->data;
		break;
	case Value_OP_MUL:
		self->data = self->prevs[0]->data * self->prevs[1]->data;
		break;
	case Value_OP_DIV:
		self->data = self->prevs[0]->data / self->prevs[1]->data;
		break;
	case Value_OP_POW_CONST:
		self->data = pow(self->prevs[0]->data, self->prevs[1]->data);
		break;
	case Value_OP_NEG:
		self->data = self->prevs[0]->data * -1;
		break;
	case Value_OP_TANH:
	{
		const double ex = exp(2 * self->prevs[0]->data);
		self->data = (ex - 1) / (ex + 1);
		break;
	}
	case Value_OP_RELU:
		self->data = (self->prevs[0]->data < 0) ? 0.0 : self->prevs[0]->data;
		break;
	}
}

void Value_backward(Value *self)
{
	switch (self->op)
	{
	case Value_OP_EMPTY:
		break;
	case Value_OP_ADD:
		self->prevs[0]->grad += self->grad;
		self->prevs[1]->grad += self->grad;
		break;
	case Value_OP_SUB:
		self->prevs[0]->grad += self->grad;
		self->prevs[1]->grad -= self->grad; //-=
		break;
	case Value_OP_MUL:
		self->prevs[0]->grad += self->prevs[1]->data * self->grad;
		self->prevs[1]->grad += self->prevs[0]->data * self->grad;
		break;
	case Value_OP_DIV:
	{
		const double b = self->prevs[1]->data;
		self->prevs[0]->grad += (1.0 / b) * self->grad;
		self->prevs[1]->grad -= (self->prevs[0]->data / (b * b)) * self->grad; //-=
		break;
	}
	case Value_OP_POW_CONST:
		self->prevs[0]->grad += self->prevs[1]->data * pow(self->prevs[0]->data, self->prevs[1]->data - 1) * self->grad;
		break;
	case Value_OP_NEG:
		self->prevs[0]->grad -= self->grad; //-=
		break;
	case Value_OP_TANH:
		self->prevs[0]->grad += (1 - (self->data * self->data)) * self->grad;
		break;
	case Value_OP_RELU:
		self->prevs[0]->grad += (self->data > 0.0) * self->grad;
		break;
	}
}

typedef struct ValueAllocator_s
{
	Value **blocks; // block = 65536x Value
	int num_blocks;

	int num_values; // total # of values
} ValueAllocator;

ValueAllocator *ValueAllocator_new(void)
{
	ValueAllocator *self = malloc(sizeof(ValueAllocator));
	self->blocks = 0;
	self->num_blocks = 0;
	self->num_values = 0;
	return self;
}
void ValueAllocator_delete(ValueAllocator *self)
{
	for (int i = 0; i < self->num_blocks; i++)
	{
		memset(self->blocks[i], 0, sizeof(Value) * 65536);
		free(self->blocks[i]);
	}
	memset(self->blocks, 0, self->num_blocks * sizeof(Value *));
	free(self->blocks);

	memset(self, 0, sizeof(ValueAllocator));
	free(self);
}
Value *ValueAllocator_alloc(ValueAllocator *self)
{
	if (self->num_values % 65536 == 0)
	{
		// resizes base
		self->num_blocks++;
		self->blocks = realloc(self->blocks, self->num_blocks * sizeof(Value *));

		// adds block
		self->blocks[self->num_blocks - 1] = malloc(sizeof(Value) * 65536);
		memset(self->blocks[self->num_blocks - 1], 0, sizeof(Value) * 65536);
	}

	Value *ret = &self->blocks[self->num_blocks - 1][self->num_values % 65536];
	self->num_values++;
	return ret;
}

Value *VA_const(ValueAllocator *allocator, const double data)
{
	Value *self = _Value_init(ValueAllocator_alloc(allocator), data, Value_OP_EMPTY);
	return self;
}
Value *VA_add(ValueAllocator *allocator, Value *a, Value *b)
{
	Value *self = _Value_init(ValueAllocator_alloc(allocator), 0, Value_OP_ADD);
	Value_setPre2(self, a, b);
	return self;
}
Value *VA_sub(ValueAllocator *allocator, Value *a, Value *b)
{
	Value *self = _Value_init(ValueAllocator_alloc(allocator), 0, Value_OP_SUB);
	Value_setPre2(self, a, b);
	return self;
}

Value *VA_mul(ValueAllocator *allocator, Value *a, Value *b)
{
	Value *self = _Value_init(ValueAllocator_alloc(allocator), 0, Value_OP_MUL);
	Value_setPre2(self, a, b);
	return self;
}

Value *VA_div(ValueAllocator *allocator, Value *a, Value *b)
{
	Value *self = _Value_init(ValueAllocator_alloc(allocator), 0, Value_OP_DIV);
	Value_setPre2(self, a, b);
	return self;
}

Value *VA_powConst(ValueAllocator *allocator, Value *a, Value *b)
{
	Value *self = _Value_init(ValueAllocator_alloc(allocator), 0, Value_OP_POW_CONST);
	Value_setPre2(self, a, b);
	return self;
}

Value *VA_neg(ValueAllocator *allocator, Value *a)
{
	Value *self = _Value_init(ValueAllocator_alloc(allocator), 0, Value_OP_NEG);
	Value_setPre(self, 0, a);
	return self;
}

Value *VA_tanh(ValueAllocator *allocator, Value *a)
{
	Value *self = _Value_init(ValueAllocator_alloc(allocator), 0, Value_OP_TANH);
	Value_setPre(self, 0, a);
	return self;
}

Value *VA_relu(ValueAllocator *allocator, Value *a)
{
	Value *self = _Value_init(ValueAllocator_alloc(allocator), 0, Value_OP_RELU);
	Value_setPre(self, 0, a);
	return self;
}
