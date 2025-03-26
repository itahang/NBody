#ifndef BODY_PART
#define BODY_PART

#include<cuda_runtime.h>


#pragma pack(push, 1)
struct Body {
	double2 position;
	double2 velocity;
	double2 acceleration;
	double2 prev_position;

	double mass;
};

#pragma pack(pop)


#endif // !BODY_PART


