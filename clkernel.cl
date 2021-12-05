/* OpenCL based simple sphere path tracer by Sam Lapere, 2016*/
/* based on smallpt by Kevin Beason */
/* http://raytracey.blogspot.com */

__constant float EPSILON = 0.00003f; /* required to compensate for limited float precision */
__constant float PI = 3.14159265359f;
__constant int SAMPLES = 1;

typedef struct Ray{
	float3 origin;
	float3 dir;
} Ray;

typedef struct Sphere{
	float radius;
	float3 pos;
	float3 color;
	float3 emission;
} Sphere;

static float get_random(unsigned int *seed0, unsigned int *seed1) {

	/* hash the seeds using bitwise AND operations and bitshifts */
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	unsigned int ires = ((*seed0) << 16) + (*seed1);

	/* use union struct to convert int to float */
	union {
		float f;
		unsigned int ui;
	} res;

	res.ui = (ires & 0x007fffff) | 0x40000000;  /* bitwise AND, bitwise OR */
	return (res.f - 2.0f) / 2.0f;
}

Ray createCamRay(const int x_coord, const int y_coord, const int width, const int height){

	float3 camera_w = (float3)(0.0f,0.0f,-1.0f);
 	float3 camera_u = cross((float3)(0.0f,1.0f,0.0f), camera_w);
  	float3 camera_v = cross(camera_w, camera_u);
	float3 origin = (float3)(0.0f,0.0f,-800.0f);
	float x_film = 3.5f*(x_coord-width/-2.0f+0.5f)/(float)width; // view x range [-w/2, w/2]
	float y_film = 3.5f*(y_coord-height/2.0f+0.5f)/(float)height; // view y range [-h/2, h/2]
	float z = 5.0f; //distance_to_film
	float3 pixel = x_film*camera_u+y_film*camera_v+z*camera_w+origin;
	float3 direction = normalize(pixel - origin);

	/* create camera ray*/
	Ray ray;
	ray.origin = origin; /* fixed camera position */
	ray.dir = direction; /* vector from camera to pixel on screen */

	return ray;
}

/* (__global Sphere* sphere, const Ray* ray) */
float intersect_sphere(const Sphere* sphere, const Ray* ray) /* version using local copy of sphere */
{
	float3 rayToCenter = sphere->pos - ray->origin;
	float b = dot(rayToCenter, ray->dir);
	float c = dot(rayToCenter, rayToCenter) - sphere->radius*sphere->radius;
	float disc = b * b - c;

	if (disc < 0.0f) return 0.0f;
	else disc = sqrt(disc);

	if ((b - disc) > EPSILON) return b - disc;
	if ((b + disc) > EPSILON) return b + disc;

	return 0.0f;
}

bool intersect_scene(__constant Sphere* spheres, const Ray* ray, float* t, int* sphere_id, const int sphere_count)
{
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	float inf = 1e20f;
	*t = inf;

	/* check if the ray intersects each sphere in the scene */
	for (int i = 0; i < sphere_count; i++)  {

		Sphere sphere = spheres[i]; /* create local copy of sphere */

		/* float hitdistance = intersect_sphere(&spheres[i], ray); */
		float hitdistance = intersect_sphere(&sphere, ray);
		/* keep track of the closest intersection and hitobject found so far */
		if (hitdistance != 0.0f && hitdistance < *t) {
			*t = hitdistance;
			*sphere_id = i;
		}
	}
	return *t < inf; /* true when ray interesects the scene */
}


/* the path tracing function */
/* computes a path (starting from the camera) with a defined number of bounces, accumulates light/color at each bounce */
/* each ray hitting a surface will be reflected in a random direction (by randomly sampling the hemisphere above the hitpoint) */
/* small optimisation: diffuse ray directions are calculated using cosine weighted importance sampling */

float3 trace(__constant Sphere* spheres, const Ray* camray, const int sphere_count, const int* seed0, const int* seed1){

	Ray ray = *camray;

	float3 accum_color = (float3)(0.0f, 0.0f, 0.0f);
	float3 mask = (float3)(1.0f, 1.0f, 1.0f);

	float3 lightpoint = (float3)(0.0f, 1.36f, 0.0f);
	
	for (int bounces = 0; bounces < 8; bounces++){

		float t;   /* distance to intersection */
		int hitsphere_id = 0; /* index of intersected sphere */

		/* if ray misses scene, return background colour */
		if (!intersect_scene(spheres, &ray, &t, &hitsphere_id, sphere_count)) {
			t = 0.5f*(ray.dir.y + 1.0f);
			return (1.0f-t)*(float3)(1.0f, 1.0f, 1.0f) + t*(float3)(0.5f, 0.7f, 1.0f);
		}

		/* else, we've got a hit! Fetch the closest hit sphere */
		Sphere hitsphere = spheres[hitsphere_id]; /* version with local copy of sphere */

		/* compute the hitpoint using the ray equation */
		float3 hitpoint = ray.origin + ray.dir * t;

		/* compute the surface normal and flip it if necessary to face the incoming ray */
		float3 normal = normalize(hitpoint - hitsphere.pos);
		float3 point_to_light = normalize(lightpoint - hitpoint);
		float diffuse = dot(normal, point_to_light);
		
		Ray shadowray;
		shadowray.origin = hitpoint;
		shadowray.dir = point_to_light;
		if (diffuse < 0 || !intersect_scene(spheres, &shadowray, &t, &hitsphere_id, sphere_count)) {
		diffuse = 0.0f;
		}
		
		float3 reflect = ray.dir + normal * (dot(normal, ray.dir) * -2.0f);
		float phong = pow(dot(point_to_light, reflect)* (diffuse > 0), 64.0f);
		accum_color += (float3)(phong, phong, phong);
		
		ray.dir = reflect;
		ray.origin = hitpoint;
	}

	return accum_color;
}

union Colour{ float c; uchar4 components;};			

__kernel void render_kernel(__constant Sphere* spheres, const int width, const int height, const int sphere_count, __global float3* output, const int hashedframenumber)
{
	unsigned int work_item_id = get_global_id(0);	/* the unique global id of the work item for the current pixel */
	unsigned int x_coord = work_item_id % width;			/* x-coordinate of the pixel */
	unsigned int y_coord = work_item_id / width;			/* y-coordinate of the pixel */

	/* seeds for random number generator */
	unsigned int seed0 = x_coord + hashedframenumber;
	unsigned int seed1 = y_coord + hashedframenumber;

	Ray camray = createCamRay(x_coord, y_coord, width, height);

	/* add the light contribution of each sample and average over all samples*/
	float3 finalcolor = (float3)(0.0f, 0.0f, 0.0f);
	float invSamples = 1.0f / SAMPLES;

	for (int i = 0; i < SAMPLES; i++)
		finalcolor += trace(spheres, &camray, sphere_count, &seed0, &seed1) * invSamples;

	finalcolor = (float3)(clamp(finalcolor.x, 0.0f, 1.0f), 
		clamp(finalcolor.y, 0.0f, 1.0f), clamp(finalcolor.z, 0.0f, 1.0f));

	union Colour fcolour;
	fcolour.components = (uchar4)(	
		(unsigned char)(finalcolor.x * 255), 
		(unsigned char)(finalcolor.y * 255),
		(unsigned char)(finalcolor.z * 255),
		1);

	/* store the pixelcolour in the output buffer */
	output[work_item_id] = (float3)(x_coord, y_coord, fcolour.c);
}
