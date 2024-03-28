// Parameters
radius = 18;                                      // radius of the sphere
length = 1.5;                                   // edge length at all points
theta = 0.7;
// Create points
Point(0) = {      0,       0,       0, length};  // center
Point(1) = { radius,       0,       0, length};  // right
Point(2) = {-radius,       0,       0, length};  // left
Point(3) = {      0,  radius,       0, 0.01*length};  // top
Point(4) = {      0, -radius,       0, length};  // bottom
Point(5) = {      0,       0,  radius, length};  // front
Point(6) = {      0,       0, -radius, length};  // back
Point(7) = {radius*Sin(theta),radius*Cos(theta), 0, length}; // top right
Point(8) = {-radius*Sin(theta),radius*Cos(theta), 0, length}; // top left
Point(9) = {0,radius*Cos(theta), radius*Sin(theta), length}; // top front
Point(10) = {0,radius*Cos(theta), -radius*Sin(theta), length}; // top back

// arcs
Circle(51) = {5, 0, 1};  // front to right
Circle(52) = {5, 0, 2};	 // front to left
Circle(59) = {5, 0, 9};	 // front to top front
Circle(93) = {9, 0, 3};	 // top front to top
Circle(54) = {5, 0, 4};  // front to bottom
Circle(28) = {2, 0, 8};  // left to top left
Circle(83) = {8, 0, 3};  // top left to top
Circle(26) = {2, 0, 6};	 // left to back
Circle(24) = {2, 0, 4};	 // left to bottom
Circle(17) = {1, 0, 7};	 // right to top right
Circle(73) = {7, 0, 3};	 // top right to top
Circle(16) = {1, 0, 6};	 // right to back
Circle(14) = {1, 0, 4};	 // right to bottom
Circle(610) = {6, 0, 10}; // back to top back
Circle(103) = {10, 0, 3}; // top back to top
Circle(64) = {6, 0, 4};	 // back to bottom
Circle(97) = {9, 0, 7};	 //top front to top right
Circle(710) = {7, 0, 10};// top right to top back
Circle(108) = {10, 0, 8};// top back to top left
Circle(89) = {8, 0, 9}; // top left to to front

// Line Loop
Line Loop(5179) = {51, 17, -97, -59};	 // front, right, top right, top front, front
Line Loop(5289) = {52, 28, 89, -59};	 // front, left, top left, top front, front
Line Loop(514) = {51, 14, -54};		 // front, right, bottom, front
Line Loop(524) = {52, 24, -54};		 // front, left, bottom, front
Line Loop(61710) = {-16, 17, 710, -610};	 // back, right, top right, top back, back
Line Loop(62810) = {-26, 28, -108, -610};	 // back, left, top left, top back, back
Line Loop(614) = {-16, 14, -64};		 // back, right, bottom, back
Line Loop(624) = {-26, 24, -64};		 // back, left, bottom, back
Line Loop(973) = {97, 73, -93};		 // top front, top right, top, top front
Line Loop(7103) = {710, 103, -73};	 // top right, top back, top, top right
Line Loop(1083) = {108, 83, -103};	 // top back, top left, top, top back
Line Loop(893) = {89, 93, -83};	 	 // top left, top front, top, top left 

Surface(1) = {+5179};
Surface(2) = {-5289};
Surface(3) = {-514};
Surface(4) = {+524};
Surface(5) = {-61710};
Surface(6) = {+62810};
Surface(7) = {+614};
Surface(8) = {-624};
Surface(9) = {+973};
Surface(10) = {+7103};
Surface(11) = {+1083};
Surface(12) = {+893};

Surface Loop(100) = {1,2,3,4,5,6,7,8,9,10,11,12};
Volume(1000) = {100};
