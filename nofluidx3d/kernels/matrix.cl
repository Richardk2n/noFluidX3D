/**
 * \file	matrix.cl
 *
 * \brief	Implementation for "class" Double3x3
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

Float3x3 INL OVL fromRows(const float3 r1, const float3 r2, const float3 r3) {
	return (Float3x3) {
		r1.x, r1.y, r1.z,
		r2.x, r2.y, r2.z,
		r3.x, r3.y, r3.z
	};
}

Float3x3 INL OVL fromColumns(const float3 c1, const float3 c2, const float3 c3) {
	return (Float3x3) {
		c1.x, c2.x, c3.x,
		c1.y, c2.y, c3.y,
		c1.z, c2.z, c3.z
	};
}

float3 INL OVL getRow(const Float3x3 matrix, const int i) {
	if(i == 1) {
		return (float3)(matrix.m11, matrix.m12, matrix.m13);
	} else if(i == 2) {
		return (float3)(matrix.m21, matrix.m22, matrix.m23);
	} else {
		return (float3)(matrix.m31, matrix.m32, matrix.m33);
	}
}

float3 INL OVL getColumn(const Float3x3 matrix, const int j) {
	if(j == 1) {
		return (float3)(matrix.m11, matrix.m21, matrix.m31);
	} else if(j == 2) {
		return (float3)(matrix.m12, matrix.m22, matrix.m32);
	} else {
		return (float3)(matrix.m13, matrix.m23, matrix.m33);
	}
}

float INL OVL Tr(const Float3x3 matrix) {
	return matrix.m11 + matrix.m22 + matrix.m33;
}

float INL OVL det(const Float3x3 m) {
	return dot(getColumn(m, 1), cross(getColumn(m, 2), getColumn(m, 3)));
}

Float3x3 INL OVL transpose(const Float3x3 m) {
	return (Float3x3) {
		m.m11, m.m21, m.m31,
		m.m12, m.m22, m.m32,
		m.m13, m.m23, m.m33
	};
}

Float3x3 INL OVL invert(const Float3x3 m) {
	// We could check for det(m) = 0 here
	return devide(fromColumns(cross(getColumn(m, 2), getColumn(m, 3)), cross(getColumn(m, 3), getColumn(m, 1)), cross(getColumn(m, 1), getColumn(m, 2))), det(m));
}

Float3x3 INL OVL multiply(const Float3x3 a, const Float3x3 b) {
	return (Float3x3) {
		dot(getRow(a, 1), getColumn(b, 1)), dot(getRow(a, 1), getColumn(b, 2)), dot(getRow(a, 1), getColumn(b, 3)),
		dot(getRow(a, 2), getColumn(b, 1)), dot(getRow(a, 2), getColumn(b, 2)), dot(getRow(a, 2), getColumn(b, 3)),
		dot(getRow(a, 3), getColumn(b, 1)), dot(getRow(a, 3), getColumn(b, 2)), dot(getRow(a, 3), getColumn(b, 3))
	};
}

float3 INL OVL multiply(const Float3x3 m, const float3 v) {
	return (float3)(dot(getRow(m, 1), v), dot(getRow(m, 2), v), dot(getRow(m, 3), v));
}


Float3x3 INL OVL multiply(const Float3x3 m, const float s) {
	return (Float3x3) {
		m.m11*s, m.m12*s, m.m13*s,
		m.m21*s, m.m22*s, m.m23*s,
		m.m31*s, m.m32*s, m.m33*s,
	};
}

Float3x3 INL OVL devide(const Float3x3 m, const float s) {
	return multiply(m, 1/s);
}

Double3x3 INL OVL fromRows(const double3 r1, const double3 r2, const double3 r3) {
	return (Double3x3) {
		r1.x, r1.y, r1.z,
		r2.x, r2.y, r2.z,
		r3.x, r3.y, r3.z
	};
}

Double3x3 INL OVL fromColumns(const double3 c1, const double3 c2, const double3 c3) {
	return (Double3x3) {
		c1.x, c2.x, c3.x,
		c1.y, c2.y, c3.y,
		c1.z, c2.z, c3.z
	};
}

double3 INL OVL getRow(const Double3x3 matrix, const int i) {
	if(i == 1) {
		return (double3)(matrix.m11, matrix.m12, matrix.m13);
	} else if(i == 2) {
		return (double3)(matrix.m21, matrix.m22, matrix.m23);
	} else {
		return (double3)(matrix.m31, matrix.m32, matrix.m33);
	}
}

double3 INL OVL getColumn(const Double3x3 matrix, const int j) {
	if(j == 1) {
		return (double3)(matrix.m11, matrix.m21, matrix.m31);
	} else if(j == 2) {
		return (double3)(matrix.m12, matrix.m22, matrix.m32);
	} else {
		return (double3)(matrix.m13, matrix.m23, matrix.m33);
	}
}

double INL OVL Tr(const Double3x3 matrix) {
	return matrix.m11 + matrix.m22 + matrix.m33;
}

double INL OVL det(const Double3x3 m) {
	return dot(getColumn(m, 1), cross(getColumn(m, 2), getColumn(m, 3)));
}

Double3x3 INL OVL transpose(const Double3x3 m) {
	return (Double3x3) {
		m.m11, m.m21, m.m31,
		m.m12, m.m22, m.m32,
		m.m13, m.m23, m.m33
	};
}

Double3x3 INL OVL invert(const Double3x3 m) {
	// We could check for det(m) = 0 here
	return devide(fromColumns(cross(getColumn(m, 2), getColumn(m, 3)), cross(getColumn(m, 3), getColumn(m, 1)), cross(getColumn(m, 1), getColumn(m, 2))), det(m));
}

Double3x3 INL OVL multiply(const Double3x3 a, const Double3x3 b) {
	return (Double3x3) {
		dot(getRow(a, 1), getColumn(b, 1)), dot(getRow(a, 1), getColumn(b, 2)), dot(getRow(a, 1), getColumn(b, 3)),
		dot(getRow(a, 2), getColumn(b, 1)), dot(getRow(a, 2), getColumn(b, 2)), dot(getRow(a, 2), getColumn(b, 3)),
		dot(getRow(a, 3), getColumn(b, 1)), dot(getRow(a, 3), getColumn(b, 2)), dot(getRow(a, 3), getColumn(b, 3))
	};
}

double3 INL OVL multiply(const Double3x3 m, const double3 v) {
	return (double3)(dot(getRow(m, 1), v), dot(getRow(m, 2), v), dot(getRow(m, 3), v));
}


Double3x3 INL OVL multiply(const Double3x3 m, const double s) {
	return (Double3x3) {
		m.m11*s, m.m12*s, m.m13*s,
		m.m21*s, m.m22*s, m.m23*s,
		m.m31*s, m.m32*s, m.m33*s,
	};
}

Double3x3 INL OVL devide(const Double3x3 m, const double s) {
	return multiply(m, 1/s);
}
