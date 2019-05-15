/**
 * \file
 * cub::DefaultMask provides HIP shuffle-mask compatibility to warp shuffle operators
 */


#pragma once
/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

#ifdef __HIP_PLATFORM_HCC__
     typedef unsigned long mask_type;
#else
     typedef unsigned int mask_type;
#endif
/**
 *  Return a default execution mask (all warp lanes enabled) with correct size for the HIP platform in use
 */
__device__ __forceinline__ mask_type DefaultMask()
{
	return ((mask_type)-1);
}

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
