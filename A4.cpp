#include <glm/ext.hpp>

#include "A4.hpp"

#include "Primitive.hpp"
#include <assert.h>
#include <algorithm>
#include <limits>

using namespace glm;

double global_t = DBL_MAX; // global t for cube face intersection
const double EPSILON = 0.0000001;
vec3 curpixelNormal = vec3(0,0,0);

vec3 unit_vector(vec3 v) {
    return glm::normalize(v);
}

double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// return random double in [0,1]
double uniform()
{
    return static_cast< double >( rand() ) / RAND_MAX;
}

bool hit_backcube_test (double xmax,double xmin,double ymax,double ymin,double zmax,double zmin, const vec3 &ray_origin, const vec3 &ray_direction) {
  vec3 n = vec3(0,0,1);
  //std::cout<<"vec n"<<to_string(n)<<std::endl;
  double t = dot(vec3(xmax, ymax, zmax) - ray_origin, n)/dot(ray_direction, n);
  //std::cout<<"double t "<<t<<std::endl;
  vec3 p = ray_origin + t*ray_direction;
  //std::cout<<"vec p "<<to_string(p)<<std::endl;
  if (p[0] > xmax + EPSILON|| p[0] < xmin - EPSILON|| p[1] > ymax + EPSILON || p[1] < ymin - EPSILON|| p[2] > zmax + EPSILON|| p[2] < zmin - EPSILON) {
    return false;
  }
  if (t < global_t) {
    curpixelNormal[0]=0;
    curpixelNormal[1]=0;
    curpixelNormal[2]=1;
    global_t = t;
  }
  return true;
  
}

bool hit_left_test (double xmax,double xmin,double ymax,double ymin,double zmax,double zmin, const vec3 &ray_origin, const vec3 &ray_direction) {
  vec3 n = vec3(1,0,0);
  //std::cout<<"vec n"<<to_string(n)<<std::endl;
  double t = dot(vec3(xmin, ymax, zmax) - ray_origin, n)/dot(ray_direction, n);
  //std::cout<<"double t "<<t<<std::endl;
  vec3 p = ray_origin + t*ray_direction;
  //std::cout<<"vec p "<<to_string(p)<<std::endl;
  if (p[0] > xmax + EPSILON|| p[0] < xmin - EPSILON|| p[1] > ymax + EPSILON || p[1] < ymin - EPSILON|| p[2] > zmax + EPSILON|| p[2] < zmin - EPSILON) {
    return false;
  }
  if (t < global_t) {
    curpixelNormal[0]=-1;
    curpixelNormal[1]=0;
    curpixelNormal[2]=0;
    global_t = t;
  }
  return true;
}

bool hit_right_test (double xmax,double xmin,double ymax,double ymin,double zmax,double zmin, const vec3 &ray_origin, const vec3 &ray_direction) {
  vec3 n = vec3(1,0,0);
  //std::cout<<"vec n"<<to_string(n)<<std::endl;
  double t = dot(vec3(xmax, ymax, zmax) - ray_origin, n)/dot(ray_direction, n);
  //std::cout<<"double t "<<t<<std::endl;
  vec3 p = ray_origin + t*ray_direction;
  //std::cout<<"vec p "<<to_string(p)<<std::endl;
  if (p[0] > xmax + EPSILON|| p[0] < xmin - EPSILON|| p[1] > ymax + EPSILON || p[1] < ymin - EPSILON|| p[2] > zmax + EPSILON|| p[2] < zmin - EPSILON) {
    return false;
  }
  if (t < global_t) {
    curpixelNormal[0]=1;
    curpixelNormal[1]=0;
    curpixelNormal[2]=0;
    global_t = t;
  }
  return true;
}

bool hit_top_test (double xmax,double xmin,double ymax,double ymin,double zmax,double zmin, const vec3 &ray_origin, const vec3 &ray_direction) {
  vec3 n = vec3(0,-1,0);
  //std::cout<<"vec n"<<to_string(n)<<std::endl;
  double t = dot(vec3(xmax, ymax, zmax) - ray_origin, n)/dot(ray_direction, n);
  //std::cout<<"double t "<<t<<std::endl;
  vec3 p = ray_origin + t*ray_direction;
  //std::cout<<"vec p "<<to_string(p)<<std::endl;
  if (p[0] > xmax + EPSILON|| p[0] < xmin - EPSILON|| p[1] > ymax + EPSILON || p[1] < ymin - EPSILON|| p[2] > zmax + EPSILON|| p[2] < zmin - EPSILON) {
    return false;
  }
  if (t < global_t) {
    curpixelNormal[0]=0;
    curpixelNormal[1]=1;
    curpixelNormal[2]=0;
    global_t = t;
  }
  return true;
}

bool hit_bottom_test (double xmax,double xmin,double ymax,double ymin,double zmax,double zmin, const vec3 &ray_origin, const vec3 &ray_direction) {
  vec3 n = vec3(0,1,0);
  //std::cout<<"vec n"<<to_string(n)<<std::endl;
  double t = dot(vec3(xmin, ymin, zmax) - ray_origin, n)/dot(ray_direction, n);
  //std::cout<<"double t "<<t<<std::endl;
  vec3 p = ray_origin + t*ray_direction;
  //std::cout<<"vec p "<<to_string(p)<<std::endl;
  if (p[0] > xmax + EPSILON|| p[0] < xmin - EPSILON|| p[1] > ymax + EPSILON || p[1] < ymin - EPSILON|| p[2] > zmax + EPSILON|| p[2] < zmin - EPSILON) {
    return false;
  }
  if (t < global_t) {
    curpixelNormal[0]=0;
    curpixelNormal[1]=-1;
    curpixelNormal[2]=0;
    global_t = t;
  }
  return true;
}

bool hit_frontcube_test (double xmax,double xmin,double ymax,double ymin,double zmax,double zmin, const vec3 &ray_origin, const vec3 &ray_direction) {
  vec3 n = vec3(0,0,-1);
  //std::cout<<"vec n"<<to_string(n)<<std::endl;
  double t = dot(vec3(xmax, ymax, zmin) - ray_origin, n)/dot(ray_direction, n);
  //std::cout<<"double t "<<t<<std::endl;
  vec3 p = ray_origin + t*ray_direction;
  //std::cout<<"vec p "<<to_string(p)<<std::endl;
  if (p[0] > xmax + EPSILON|| p[0] < xmin - EPSILON|| p[1] > ymax + EPSILON || p[1] < ymin - EPSILON|| p[2] > zmax + EPSILON|| p[2] < zmin - EPSILON) {
    return false;
  }
  if (t < global_t) {
    curpixelNormal[0]=0;
    curpixelNormal[1]=0;
    curpixelNormal[2]=-1;
    global_t = t;
  }
  return true;
  
}

bool hit_cube_test (double xmax,double xmin,double ymax,double ymin,double zmax,double zmin, const vec3 &ray_origin, const vec3 &ray_direction, vec3 &normal_out) {
  double global_t = DBL_MAX;
  curpixelNormal = vec3(0,0,0);
  auto b1 = hit_frontcube_test( xmax, xmin, 
                  ymax, ymin, 
                  zmax, zmin, 
                  ray_origin, ray_direction);
  auto b2 = hit_backcube_test ( xmax, xmin, 
                  ymax, ymin, 
                  zmax, zmin, 
                  ray_origin, ray_direction);
  auto b3 = hit_left_test( xmax, xmin, 
                  ymax, ymin, 
                  zmax, zmin, 
                  ray_origin, ray_direction); 
  auto b4 = hit_right_test ( xmax, xmin, 
                  ymax, ymin, 
                  zmax, zmin, 
                  ray_origin, ray_direction); 
  auto b5 = hit_top_test( xmax, xmin, 
                  ymax, ymin, 
                  zmax, zmin, 
                  ray_origin, ray_direction);
  auto b6 = hit_bottom_test(  xmax, xmin, 
                  ymax, ymin, 
                  zmax, zmin, 
                  ray_origin, ray_direction);
                  
  normal_out[0] = curpixelNormal[0];
  normal_out[1] = curpixelNormal[1];
  normal_out[2] = curpixelNormal[2];
  return (b1 || b2 || b3 || b4 || b5 || b6 );
}

bool slab_test (double xmax,double xmin,double ymax,double ymin,double zmax,double zmin, const vec3 &ray_origin, const vec3 &ray_direction) {
  
  double txmax = (xmax - ray_origin[0])/ray_direction[0];
  double txmin = (xmin - ray_origin[0])/ray_direction[0];
  
  if (txmin > txmax) std::swap(txmin, txmax); 
  
  double tymin = (ymax - ray_origin[1])/ray_direction[1];
  double tymax = (ymin - ray_origin[1])/ray_direction[1];
  
  if (tymin > tymax) std::swap(tymin, tymax); 
  
  if ((txmin > tymax) || (tymin > txmax))
      return false;

  if (tymin > txmin)
      txmin = tymin;
  if (tymax < txmax)
      txmax = tymax;
  
  double tzmax = (ymax - ray_origin[2])/ray_direction[2];
  double tzmin = (ymin - ray_origin[2])/ray_direction[2];
  
  double tmin = std::max(std::min(txmax, txmin), min(tymin, tymax));
  double tmax = std::min(std::max(txmax, txmin), max(tymin, tymax));
  return (tmin < tmax && (tmin > 0 || tmax > 0));
  
  if (tzmin > tzmax) std::swap(tzmin, tzmax);
  
  if ((txmin > tzmax) || (tzmin > txmax))
      return false;
  if (tzmin > txmin)
      txmin = tzmin;
  if (tzmax < txmax)
      txmax = tzmax;

  return true; 
  
}

/* adapt from Möller Trumbore ray-triangle intersection algorithm
 * https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
 */
bool RayIntersectsTriangle(vec3 rayOrigin, 
                           vec3 rayVector, 
                           vec3 Trianglev0, //triangle vertex0
                           vec3 Trianglev1,
                           vec3 Trianglev2,
                           vec3 &outIntersectionPoint,
                           double &t)
{

    vec3 edge1, edge2, h, s, q, curNormal;
    double a,f,u,v;
    edge1 = Trianglev1 - Trianglev0;
    edge2 = Trianglev2 - Trianglev0;
    
    curNormal = unit_vector(cross(edge1, edge2));
    
    h = cross(rayVector, edge2);
    a = dot(edge1,h); //determinant of |-rayVector, edge1, edge2|
    if (a > -EPSILON && a < EPSILON) // determinant is very small
        return false;
    f = 1.0/a;
    s = rayOrigin - Trianglev0;
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;
    q = cross(s, edge1);
    v = f * dot(rayVector, q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    // compute t to find out where the intersection point is.
    t = f * dot(edge2, q);
    if (t > EPSILON) // ray intersection
    {
        if (t < global_t) global_t = t;
        curpixelNormal = curNormal;
        outIntersectionPoint = rayOrigin + rayVector * t;
        return true;
    }
    else // there is a line intersection, not a ray intersection.
        return false;
}


double ray_hitsphere(const vec3 &center, double radius, const vec3 &ray_origin, const vec3 &ray_direction) {
    vec3 o_minus_c = ray_origin - center;
    auto a = dot(ray_direction, ray_direction);
    auto b = 2.0 * dot(o_minus_c, ray_direction);
    auto c = dot(o_minus_c, o_minus_c) - radius*radius;
    auto discriminant = b*b - 4*a*c;
    if (discriminant < 0) {
      return DBL_MAX;
    } else if (discriminant == 0) {
       return (-b + sqrt(discriminant) ) / (2.0*a);
    } else {
      if ((-b - sqrt(discriminant) ) / (2.0*a) < 0 && (-b + sqrt(discriminant) ) / (2.0*a) > 0) {
        //std::cout<<"origin is in the sphere"<<std::endl;
        return (-b + sqrt(discriminant) ) / (2.0*a);
      }
      else if ((-b - sqrt(discriminant) ) / (2.0*a) > 0 && (-b + sqrt(discriminant) ) / (2.0*a) < 0) {
        //std::cout<<"origin is in the sphere"<<std::endl;
        return (-b - sqrt(discriminant) ) / (2.0*a);
      }
      else if ((-b - sqrt(discriminant) ) / (2.0*a) < 0 && (-b + sqrt(discriminant) ) / (2.0*a) < 0 ) {
        //std::cout<<"hit points are behind"<<std::endl;
        //return (-b + sqrt(discriminant) ) / (2.0*a);
        return DBL_MAX;
      }else { // both solutions are positive, return smaller one
        return std::min( (-b - sqrt(discriminant)) / (2.0*a), ( -b + sqrt(discriminant) ) / (2.0*a));
      }
    }
}

vec3 ray_color(const vec3 &ray_origin, const vec3 &ray_direction) {
    vec3 unit_direction = unit_vector(ray_direction);
    auto t = 0.5*(ray_direction[1] + 1.0);
    return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}


/*
 * return positive t for current pixel ray
 * if not hit, return t is DBL_MAX
 * identify primitives by their scenenode name[0]
 */ 
double forobjects_wrapper_recursive (SceneNode * root, vec3 &origin, vec3 &direction, std::string exclude_name, std::string &hit_obj_name, vec3 &curnormal) {
  double ret_t = DBL_MAX;
  hit_obj_name = "";
  
  if (root->m_nodeType == NodeType::GeometryNode && root->m_name.compare(exclude_name) != 0) {
      GeometryNode * curgeonode = dynamic_cast<GeometryNode *>(root);
      
      if (curgeonode->m_name[0] == 's') {
        
        vec3 curnode_position = dynamic_cast<NonhierSphere *>(curgeonode->m_primitive)->m_pos;
        double cursphere_radius = dynamic_cast<NonhierSphere *>(curgeonode->m_primitive)->m_radius;
        ret_t = ray_hitsphere(curnode_position, cursphere_radius, origin, direction);
        curpixelNormal = unit_vector(origin+direction*ret_t - curnode_position);
        
      } else if (curgeonode->m_name[0] == 'b') {
        
        global_t = DBL_MAX;
        vec3 curnode_position = dynamic_cast<NonhierBox *>(curgeonode->m_primitive)->m_pos;
        double curnode_size = dynamic_cast<NonhierBox *>(curgeonode->m_primitive)->m_size;
        vec3 out_normal;
        if (hit_cube_test(curnode_position[0]+curnode_size, curnode_position[0], 
                  curnode_position[1]+curnode_size, curnode_position[1], 
                  curnode_position[2]+curnode_size, curnode_position[2], 
                  origin, direction, out_normal)) {
          ret_t = global_t;
          global_t = DBL_MAX;
          curnormal = out_normal;
        }
        
      } else if (curgeonode->m_name[0] == 't') {
        
        vec3 tv1 = dynamic_cast<MyTriangle *>(curgeonode->m_primitive)->v1;
        vec3 tv2 = dynamic_cast<MyTriangle *>(curgeonode->m_primitive)->v2;
        vec3 tv3 = dynamic_cast<MyTriangle *>(curgeonode->m_primitive)->v3;
        vec3 outpoint;
        double tri_t = DBL_MAX;
        
        if (RayIntersectsTriangle(origin, direction, tv1, tv2, tv3, outpoint, tri_t)) {
          ret_t = tri_t;
        }
        
      }
  }
  
  if (ret_t != DBL_MAX) {hit_obj_name = root->m_name; curnormal = curpixelNormal;}
  
  double children_t = DBL_MAX;
  vec3 cur_children_normal; //closest point's normal for all children
  std::string global_chidren_hitname = "";
  for (auto cur_node : root->children) {
    vec3 tmp_normal;
    global_t = DBL_MAX;
    std::string cur_chidren_hitname = "";
    double cur_t = forobjects_wrapper_recursive(cur_node, origin, direction, exclude_name, cur_chidren_hitname, tmp_normal);
    if (cur_t < children_t) {children_t = cur_t; cur_children_normal = tmp_normal; global_chidren_hitname = cur_chidren_hitname;}
  }
  if (children_t < ret_t) {ret_t = children_t; hit_obj_name = global_chidren_hitname; curnormal = cur_children_normal;}
  return ret_t;
}

// compute the light intensity received by ray from given direction
vec3 shade(SceneNode * root, vec3 origin, vec3 direction, Light * curlight, int depth, std::string excludeobj) {
  vec3 curnormal;
  std::string cur_hitname;
  double cur_t = forobjects_wrapper_recursive(root, origin, direction, excludeobj, cur_hitname, curnormal);
  
  global_t = DBL_MAX;
  if ( cur_t == DBL_MAX ) { // hit the sky
    double t = 0.5*(direction[1] + 1.0);
    return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);

  } else {
      // ray hit point
      vec3 intersection_point = origin+direction*cur_t;
      // normalized direction from hit to light source
      vec3 point_to_light = unit_vector(curlight->position - intersection_point );
      
      vec3 N = curnormal;
      // Lambertial reflectance, do not depend on the camera's position
      double diffuse = dot(N, point_to_light);
      
      std::string shadow_hitname;
      vec3 normal_tmp;
      // clamp diffuse if negative or the shadow ray hit something, then clear the diffuse value
      if ( diffuse < 0 || forobjects_wrapper_recursive(root, intersection_point, point_to_light, cur_hitname, shadow_hitname, normal_tmp) != DBL_MAX ) {
        diffuse = 0.0;
      }
      
      // reflect ray direction
      vec3 reflect = direction + curnormal * (dot(curnormal, direction) * -2.0);
      // phong effect for specular highlight, larger the power to sharper the highlights
      // Specular reflectance should be calculated only for surfaces oriented to the light source
      // that means dot(Normal, PtoLight) >0
      double phong = pow(dot(point_to_light, reflect)* (diffuse > 0), 64.0);
      
      // recursively invoke shade to compute the light intensity from point coming from reflection direction
      ++depth;
      if (depth > 2) {
        // return the sky's color if depth is more than 2
        double t = 0.5*(reflect[1] + 1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
      } else {
        //double LambertianLight = (diffuse+1.0)/2.0;
        //vec3 cur_color = LambertianLight*curlight->colour + vec3(phong, phong, phong);
        //return cur_color;
        return shade(root, intersection_point, reflect, curlight, depth, cur_hitname) *0.5 + vec3(phong, phong, phong);
      }
  }
}

void A4_Render(
		// What to render  
		SceneNode * root,

		// Image to write to, set to a given width and height  
		Image & image,

		// Viewing parameters  
		const vec3 & eye,
		const vec3 & view,
		const vec3 & up,
		double fovy,

		// Lighting parameters  
		const vec3 & ambient,
		const std::list<Light *> & lights
) {
  
  double film_h = 3.5; //film height
  double film_w = 3.5; // film width
  uint res_x = image.width(); // pic resolution x
  uint res_y = image.height(); // pic resolution y
  double distance_to_film = 5; // object depth relative to film
  
  /* origin vector e for camera view point and camera ray from point*/
  vec3 origin  = vec3(0,0,0);

  vec3 camera_w = vec3(0,0,-1);
  vec3 camera_u = cross(vec3(0,1,0), camera_w);
  vec3 camera_v = cross(camera_w, camera_u);
  
  assert(dot(camera_w, camera_u) == 0);
  assert(dot(camera_w, camera_v) == 0);
  assert(dot(camera_v, camera_u) == 0);
  assert(length(camera_w) == 1);
  assert(length(camera_u) == 1);
  assert(length(camera_v) == 1);


  std::cout << "Calling A4_Render(\n" <<
		  "\t" << *root <<
          "\t" << "Image(width:" << image.width() << ", height:" << image.height() << ")\n"
          "\t" << "eye:  " << to_string(eye) << std::endl <<
		  "\t" << "view: " << to_string(view) << std::endl <<
		  "\t" << "up:   " << to_string(up) << std::endl <<
		  "\t" << "fovy: " << fovy << std::endl <<
          "\t" << "ambient: " << to_string(ambient) << std::endl <<
		  "\t" << "lights{" << std::endl;

	for(const Light * light : lights) {
		std::cout << "\t\t" <<  *light << std::endl;
	}
	std::cout << "\t}" << std::endl;
	std:: cout <<")" << std::endl;

	size_t h = image.height();
	size_t w = image.width();
  
  
  origin = vec3 (0,0,-800);
	for (uint y = 0; y < h; ++y) {
		for (uint x = 0; x < w; ++x) {
			
			double x_film = film_w*(x-w/2.0+0.5)/res_x; // view x range [-w/2, w/2]
			double y_film = film_h*(y-h/2.0+0.5)/res_y; // view y range [-h/2, h/2]
			double z = distance_to_film;
			vec3 pixel = x_film*camera_u+y_film*camera_v+z*camera_w+origin;
			vec3 direction = normalize(origin - pixel);
      
			vec3 cur_color;
      
      vec3 intersection_point;
      double triangle_t = 0;
      global_t = DBL_MAX; // smallest t for current pixel camera ray direction
      curpixelNormal[0]=0;
      curpixelNormal[1]=0;
      curpixelNormal[2]=0;
      
      std::string cur_hitname;
      vec3 normal_forpixel = vec3();
      
      vec3 randvec = vec3(uniform()-uniform(), uniform()-uniform(), uniform()-uniform());
      
      vec3 rand_direction = direction + vec3(uniform()/512.0);
      
      std::string tmpex = "f";
      cur_color = shade(root, origin, direction, lights.front(), 0, tmpex);
      cur_color[0] = clamp(cur_color[0], 0.0, 0.999);
      cur_color[1] = clamp(cur_color[1], 0.0, 0.999);
      cur_color[2] = clamp(cur_color[2], 0.0, 0.999);
      /*
      cur_color += shade(root, origin, rand_direction, lights.front(), 0);
      cur_color += shade(root, origin, rand_direction, lights.front(), 0);
      cur_color += shade(root, origin, rand_direction, lights.front(), 0);
      cur_color += shade(root, origin, rand_direction, lights.front(), 0);
      
      cur_color /= 4.0;
      
      cur_color[0] = clamp(cur_color[0], 0.0, 0.999);
      cur_color[1] = clamp(cur_color[1], 0.0, 0.999);
      cur_color[2] = clamp(cur_color[2], 0.0, 0.999);
      */ 
      /*
      double cur_t = forobjects_wrapper_recursive(root, origin, direction, "placeholder", cur_hitname, normal_forpixel);
      vec3 curpixelNormal_bak = curpixelNormal;
      
      global_t = DBL_MAX;
      vec3 out_normal = vec3(0,0,0);
      if ( cur_t == DBL_MAX ) {

				cur_color = vec3(0,0,1);

      } else {
          vec3 intersection_point = origin+direction*cur_t;
          vec3 point_to_light = unit_vector(lights.front()->position - intersection_point);
          
          std::string cur_hitshadowray_name;
          
          std::string cur_tmpname; //object hit by shadow ray 
          
          global_t = DBL_MAX;
          vec3 curpixelNormal_tmp;
          double no_ground_t = forobjects_wrapper_recursive(root, intersection_point, point_to_light, "s3", cur_tmpname, curpixelNormal_tmp);
          
          //&& no_ground_t != DBL_MAX
          //&& cur_tmpname == "b1"
          if (cur_hitname == "s3" && cur_tmpname != "s3" && no_ground_t != DBL_MAX) {
            //std::cout<<"cur_tmpname: "<<cur_tmpname<<std::endl;
            cur_color = vec3(0,0,0);
          } else {
            //if (cur_hitname == "t1" ) {
            //  std::cout<<"normal_forpixel: "<<to_string(normal_forpixel)<<std::endl;
            //}
            vec3 N = normal_forpixel;
            // Lambertial reflectance, do not depend on the camera's position
            double diffuse = dot(N, point_to_light);
            
            // reflect ray direction
            vec3 reflect = direction + normal_forpixel * (dot(normal_forpixel, direction) * -2.0 );
            // phong effect for specular highlight, larger the power to sharper the highlights
            // Specular reflectance should be calculated only for surfaces oriented to the light source
            // that is dot(Normal, PtoLight) >0
            double phong = pow(dot(point_to_light, reflect)* (diffuse > 0), 64.0);
            double LambertianLight = (diffuse+1.0)/2.0;
            cur_color = LambertianLight*lights.front()->colour + vec3(phong, phong, phong);
            
          }
      }
      */
			
			// Red: 
			image(x, y, 0) = cur_color[0];
			// Green: 
			image(x, y, 1) = cur_color[1];
			// Blue: 
			image(x, y, 2) = cur_color[2];
		}
	}
}
