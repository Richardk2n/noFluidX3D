#pragma once
/**
 * \file	matrix.h
 *
 * \brief	Header for "class" Double3x3
 *
 * \details	Provides the matrix types OpenCL should have had.
 * 			Used for force calculations in material models.
 * 			ATTENTION All indices are 1 based!
 *
 * \ingroup	opencl
 *
 * \author	Richard Kellnberger
 * \date	2025-06-16
 */

#define INL __attribute__((always_inline))
#define OVL __attribute__((overloadable))

typedef struct Float3x3_t {
	float m11, m12, m13;
	float m21, m22, m23;
	float m31, m32, m33;
} Float3x3;

Float3x3 INL OVL fromRows(const float3 r1, const float3 r2, const float3 r3);
Float3x3 INL OVL fromColumns(const float3 c1, const float3 c2, const float3 c3);

float3 INL OVL getRow(const Float3x3 matrix, const int i);
float3 INL OVL getColumn(const Float3x3 matrix, const int j);

float INL OVL Tr(const Float3x3 matrix);
float INL OVL det(const Float3x3 m);
Float3x3 INL OVL transpose(const Float3x3 m);
Float3x3 INL OVL invert(const Float3x3 m);

Float3x3 INL OVL multiply(const Float3x3 a, const Float3x3 b);
float3 INL OVL multiply(const Float3x3 m, const float3 v);
Float3x3 INL OVL multiply(const Float3x3 m, const float s);
Float3x3 INL OVL devide(const Float3x3 m, const float s);

typedef struct Double3x3_t {
	double m11, m12, m13;
	double m21, m22, m23;
	double m31, m32, m33;
} Double3x3;

Double3x3 INL OVL fromRows(const double3 r1, const double3 r2, const double3 r3);
Double3x3 INL OVL fromColumns(const double3 c1, const double3 c2, const double3 c3);

double3 INL OVL getRow(const Double3x3 matrix, const int i);
double3 INL OVL getColumn(const Double3x3 matrix, const int j);

double INL OVL Tr(const Double3x3 matrix);
double INL OVL det(const Double3x3 m);
Double3x3 INL OVL transpose(const Double3x3 m);
Double3x3 INL OVL invert(const Double3x3 m);

Double3x3 INL OVL multiply(const Double3x3 a, const Double3x3 b);
double3 INL OVL multiply(const Double3x3 m, const double3 v);
Double3x3 INL OVL multiply(const Double3x3 m, const double s);
Double3x3 INL OVL devide(const Double3x3 m, const double s);

#define unitTensor(type) (type) {1, 0, 0,\
								 0, 1, 0,\
								 0, 0, 1}
