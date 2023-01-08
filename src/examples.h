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

#define NUMBER_OF_THREADS -1 // max CPU usage

void example1(void)
{
	ValueAllocator *va = ValueAllocator_new();

	Value *a = VA_const(va, -2);
	Value *b = VA_const(va, 3);

	Value *e = VA_add(va, a, b);
	Value *d = VA_mul(va, a, b);

	Value *f = VA_mul(va, e, d);

	// run it
	Topo *topo = Topo_new(f);
	TopoMT *topoParalel = TopoMT_new(NUMBER_OF_THREADS);
	TopoMT_run(topoParalel, topo);
	TopoMT_delete(topoParalel);
	Topo_delete(topo);

	printf("a: %f | %f\n", a->data, a->grad);
	printf("b: %f | %f\n", b->data, b->grad);
	printf("e: %f | %f\n", e->data, e->grad);
	printf("d: %f | %f\n", d->data, d->grad);
	printf("f: %f | %f\n", f->data, f->grad);

	ValueAllocator_delete(va);
}

void example2(void)
{
	ValueAllocator *va = ValueAllocator_new();

	Value *a = VA_const(va, -4); // a = -4
	Value *b = VA_const(va, 2);	 // b = 2

	Value *c = VA_add(va, a, b);												  // c = a + b	= -2
	Value *d = VA_add(va, VA_mul(va, a, b), VA_powConst(va, b, VA_const(va, 3))); // d = a * b + b^3 = 0

	c = VA_add(va, c, VA_add(va, c, VA_const(va, 1)));							  // c += c + 1 = -3
	c = VA_add(va, c, VA_add(va, VA_add(va, VA_const(va, 1), c), VA_neg(va, a))); // c += 1 + c + (-a) = -1

	d = VA_add(va, d, VA_add(va, VA_mul(va, d, VA_const(va, 2)), VA_relu(va, VA_add(va, b, a)))); // d += d * 2 + (b + a).relu()
	d = VA_add(va, d, VA_add(va, VA_mul(va, VA_const(va, 3), d), VA_relu(va, VA_sub(va, b, a)))); // d += 3 * d + (b - a).relu()

	Value *e = VA_sub(va, c, d);					// e = c - d
	Value *f = VA_powConst(va, e, VA_const(va, 2)); // f = e^2

	Value *g = VA_div(va, f, VA_const(va, 2));			// g = f / 2.0
	g = VA_add(va, g, VA_div(va, VA_const(va, 10), f)); // g += 10.0 / f

	Topo *topo = Topo_new(g);
	TopoMT *topoParalel = TopoMT_new(NUMBER_OF_THREADS);
	TopoMT_run(topoParalel, topo); // Topo_run(topo);
	TopoMT_delete(topoParalel);
	Topo_delete(topo);

	printf("g.data: %.4f == 24.7041\n", g->data);
	printf("a.grad: %.4f == 138.8338\n", a->grad);
	printf("b.grad: %.4f == 645.5773\n", b->grad);

	ValueAllocator_delete(va);
}

void example3(void)
{
	ValueAllocator *va = ValueAllocator_new();

	Value *x[] = {VA_const(va, 2), VA_const(va, 3), VA_const(va, -1)};

	const int ioSizes[] = {4, 4, 1};
	MLP *mlp = MLP_new(3, ioSizes, 3, va);

	Value **ret = MLP_build(mlp, x, va);

	Topo *topo = Topo_new(ret[0]);
	TopoMT *topoParalel = TopoMT_new(NUMBER_OF_THREADS);
	TopoMT_run(topoParalel, topo);
	// Topo_run(topo);
	TopoMT_delete(topoParalel);
	Topo_delete(topo);

	printf("ret: %f | %f\n", ret[0]->data, ret[0]->grad);

	MLP_delete(mlp);
	ValueAllocator_delete(va);
}

void example4(void)
{
	ValueAllocator *va = ValueAllocator_new();

	// inputs
	const int xs_n = 4;
	Value *xs[][3] = {{VA_const(va, 2), VA_const(va, 3), VA_const(va, -1)},
					  {VA_const(va, 3), VA_const(va, -1), VA_const(va, 0.5)},
					  {VA_const(va, 0.5), VA_const(va, 1), VA_const(va, 1)},
					  {VA_const(va, 1), VA_const(va, 1), VA_const(va, -1)}};

	// desire outputs
	Value *ys[] = {VA_const(va, 1), VA_const(va, -1), VA_const(va, -1), VA_const(va, 1)};

	// inits MLP
	const int ioSizes[] = {4, 4, 1};
	MLP *mlp = MLP_new(3, ioSizes, 3, va);

	TopoMT *topoParalel = TopoMT_new(NUMBER_OF_THREADS);

	// builds MLP topo
	Topo *topoMLP[xs_n];
	Value *ypred[xs_n];
	for (int j = 0; j < xs_n; j++)
	{
		ypred[j] = MLP_build(mlp, xs[j], va)[0];
		topoMLP[j] = Topo_new(ypred[j]);
	}

	// builds Loss TOPO
	Value *loss = VA_const(va, 0);
	for (int j = 0; j < xs_n; j++) // sum
	{
		Value *sub = VA_sub(va, ypred[j], ys[j]);
		Value *pw = VA_powConst(va, sub, VA_const(va, 2));
		loss = VA_add(va, loss, pw);
	}
	Topo *topoLoss = Topo_new(loss);

	//for (int j = 0; j < xs_n; j++)
	//	Topo_print(topoMLP[j]);
	//Topo_print(topoLoss);

	// trains network
	double st = Os_time();
	const int kkN = 20;
	for (int kk = 0; kk < kkN; kk++)
	{
		// forward pass
		for (int j = 0; j < xs_n; j++)
			TopoMT_run(topoParalel, topoMLP[j]); // Topo_run(topoMLP[j]);

		// backward pass
		TopoMT_run(topoParalel, topoLoss); // Topo_run(topoLoss);

		// update
		for (int j = 0; j < xs_n; j++)
			Topo_update(topoMLP[j], -0.05); //-

		printf("[%d] ret: %f\n", kk, loss->data);
	}

	for (int j = 0; j < xs_n; j++)
		printf("ypred: %f\n", ypred[j]->data);

	printf("Trained in %fs\n", Os_time() - st);

	// cleaning
	TopoMT_delete(topoParalel);
	Topo_delete(topoLoss);
	for (int j = 0; j < xs_n; j++)
		Topo_delete(topoMLP[j]);
	MLP_delete(mlp);
	ValueAllocator_delete(va);
}
