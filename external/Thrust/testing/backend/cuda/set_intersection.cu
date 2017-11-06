#include "hip/hip_runtime.h"
#include <unittest/unittest.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4>
__global__
void set_intersection_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1,
                             Iterator2 first2, Iterator2 last2,
                             Iterator3 result1,
                             Iterator4 result2)
{
  *result2 = thrust::set_intersection(exec, first1, last1, first2, last2, result1);
}


template<typename ExecutionPolicy>
void TestSetIntersectionDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::iterator Iterator;

  Vector a(3), b(4);

  a[0] = 0; a[1] = 2; a[2] = 4;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4;

  Vector ref(2);
  ref[0] = 0; ref[1] = 4;

  Vector result(2);
  thrust::device_vector<Iterator> end_vec(1);

  hipLaunchKernelGGL(HIP_KERNEL_NAME(set_intersection_kernel), dim3(1), dim3(1), 0, 0, exec, a.begin(), a.end(), b.begin(), b.end(), result.begin(), end_vec.begin());
  Iterator end = end_vec.front();

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}


void TestSetIntersectionDeviceSeq()
{
  TestSetIntersectionDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSetIntersectionDeviceSeq);


void TestSetIntersectionDeviceDevice()
{
  TestSetIntersectionDevice(thrust::device);
}
DECLARE_UNITTEST(TestSetIntersectionDeviceDevice);


void TestSetIntersectionCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::iterator Iterator;

  Vector a(3), b(4);

  a[0] = 0; a[1] = 2; a[2] = 4;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4;

  Vector ref(2);
  ref[0] = 0; ref[1] = 4;

  Vector result(2);

  hipStream_t s;
  hipStreamCreate(&s);

  Iterator end = thrust::set_intersection(thrust::cuda::par.on(s),
                                          a.begin(), a.end(),
                                          b.begin(), b.end(),
                                          result.begin());
  hipStreamSynchronize(s);

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);

  hipStreamDestroy(s);
}
DECLARE_UNITTEST(TestSetIntersectionCudaStreams);

