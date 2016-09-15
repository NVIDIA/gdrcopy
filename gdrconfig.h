#pragma once

#if defined __GNUC__
#if defined(__powerpc__)
#define GDRAPI_POWER
#elif defined(__i386__)
#define GDRAPI_X86
#else
#endif // arch
#endif // __GNUC__
