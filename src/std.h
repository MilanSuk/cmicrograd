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

char Std_random(size_t bytes, void *data)
{
	char ok;
#if defined(_WIN32)
	NTSTATUS res = BCryptGenRandom(NULL, data, bytes, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
	ok = (res == STATUS_SUCCESS && bytes <= ULONG_MAX);
#elif defined(__linux__) || defined(__FreeBSD__)
	ssize_t res = getrandom(data, bytes, 0);
	ok = (res >= 0 && (size_t)res == bytes);
#elif defined(__APPLE__) || defined(__OpenBSD__)
	int res = getentropy(data, bytes);
	ok = (res == 0)
#endif
	return ok;
}

double Std_random11(void) // returns number in range <-1, 1>
{
	int value;
	Std_random(4, &value);
	return value / 2147483647.0;
}

int Std_numberOfThreads(void)
{
	unsigned int ncores = 0, nthreads = 0;
	asm volatile("cpuid"
				 : "=a"(ncores), "=b"(nthreads)
				 : "a"(0xb), "c"(0x1)
				 :);
	return nthreads;
}

int Std_bmin(const int a, const int b)
{
	return a < b ? a : b;
}

int Std_bmax(const int a, const int b)
{
	return a > b ? a : b;
}

void Std_sleep(const unsigned int ms)
{
#ifdef _WIN32
	Sleep(ms); // mili-seconds
#else
	usleep(ms * 1000); // micro-seconds
#endif
}

double Os_time(void)
{
#ifdef _WIN32
	LARGE_INTEGER frequency;
	LARGE_INTEGER time;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&time);
	return ((double)time.QuadPart) / frequency.QuadPart;
#else
	struct timeval startTime;
	gettimeofday(&startTime, 0);
	return (double)startTime.tv_sec + ((double)startTime.tv_usec) / 1000000;
#endif
}

typedef struct StdThread_s
{
#ifdef WIN32
	void *thread;
#else
	unsigned long int thread;
#endif
	volatile int run;
} StdThread;

#ifdef WIN32
typedef unsigned long(__stdcall *StdThread_loopFUNC)(void *);
#define StdThread_FUNC(NAME, PARAM) unsigned long NAME(void *PARAM)
#else
typedef void *(*StdThread_loopFUNC)(void *);
#define StdThread_FUNC(NAME, PARAM) void *NAME(void *PARAM)
#endif

char StdThread_init(StdThread *self, const char *threadName, StdThread_loopFUNC func, void *funcPrm)
{
	memset(self, 0, sizeof(*self));
	self->run = 1;

#ifdef WIN32
	if (!(self->thread = CreateThread(NULL, 0, func, funcPrm, 0, NULL)))
#else
	if (pthread_create(&self->thread, 0, func, funcPrm) != 0)
#endif
	{
		memset(self, 0, sizeof(StdThread));
		return 0;
	}

	return 1;
}

char StdThread_close(StdThread *self)
{
	self->run = 0;
	if (self->thread)
#ifdef WIN32
		WaitForSingleObject(self->thread, INFINITE);
#else
		if (pthread_join(self->thread, NULL))
			return 0;
#endif
	return 1;
}

char StdThread_free(StdThread *self)
{
	return StdThread_close(self);
}

typedef struct OsSemaphore_s
{
	void *sem;
} OsSemaphore;

char OsSemaphore_init(OsSemaphore *self)
{
	self->sem = malloc(sizeof(sem_t));
	if (sem_init(self->sem, 0, 0) < 0)
		return 0;
	return 1;
}
void OsSemaphore_free(OsSemaphore *self)
{
	sem_destroy(self->sem);
	free(self->sem);
}

char OsSemaphore_trigger(OsSemaphore *self)
{
	return sem_post(self->sem) == 0;
}

char OsSemaphore_wait(OsSemaphore *self)
{
	return sem_wait(self->sem) == 0;
}
