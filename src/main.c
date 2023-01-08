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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/random.h>
#include <unistd.h>
#include <sys/time.h>

#include "std.h"
#include "value.h"
#include "topo.h"
#include "topo_mt.h"
#include "mlp.h"

#include "examples.h"

int main(int argc, char **argv)
{
	printf("---Example 1---\n");
	example1();

	printf("\n---Example 2---\n");
	example2();

	printf("\n---Example 3---\n");
	example3();

	printf("\n---Example 4---\n");
	example4();

	return 0;
}
