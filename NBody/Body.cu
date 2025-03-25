#ifndef BODY_PART
#define BODY_PART

#include<cuda_runtime.h>


#pragma pack(push, 1)
struct Body {
	float2 position;
	float2 velocity;
	float2 acceleration;
	float mass;
};

#pragma pack(pop)


#endif // !BODY_PART


