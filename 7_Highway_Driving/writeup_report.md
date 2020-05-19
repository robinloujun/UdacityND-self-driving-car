# **Highway Driving - Path Planning** 

## Writeup

---

**Highway Driving**

In this project, your goal is to design a path planner that is able to create smooth, safe paths for the car to follow along a 3 lane highway with traffic. A successful path planner will be able to keep inside its lane, avoid hitting other cars, and pass slower moving traffic all by using localization, sensor fusion, and map data.

The goals / steps of this project are the following:
* Design a path planner that is able to create smooth, safe paths
* Summarize the results with a written report

The successful path planner is able to:
- keep inside its lane
- avoid hitting other cars
- pass slower moving traffic

[//]: # (Video Reference)

[video1]: ./video_acc.mp4 "Accelerated record for 4.32 miles"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1971/view) individually and describe how I addressed each point in my implementation.  

---

### Valid Trajectories 

### 1. The car drives according to the speed limit

A variable for reference velocity is defined in main.cpp code line 57. The original value is zero and it will increase when it is under the speed limit and decrease if it reach 49.5 or needed by the traffic situation (code lines 185 - 201). So the car doesn't drive faster than the speed limit 50 mph. 

### 2. Max Acceleration and Jerk are not Exceeded

It is required that the car does not exceed a total acceleration of 10 m/s^2 and a jerk of 10 m/s^3. Since the principle of path planner is to save the trajectory points as a vector for the following frames. This can be achieved by set the proper increment between the points. In the implementation it is set the increment of velocity by 0.224.

### 3. Car does not have collisions

The car must not come into contact with any of the other cars on the road. With the given sensor fusion information, the planner can learn whether there is car so near that a collision could happen. The implementation is in main.cpp code lines 133 - 182. If there is vehicle ahead in the same lane and it drives too slowly and the distance is too close, then the ego-vehicle should brake up until it is safe again. Besides, when there are vehicles in the consecutive line then the ego-vehicle should not change the lane. 

### 4. The car stays in its lane, except for the time between changing lanes.

By using the Frenet coordinates the implementation of keeping lane is not so difficult. In the planner, three points are evenly placed ahead of the starting reference (main.cpp code lines 242 - 251), between the points we use the spline interpolation to get the trajectory points. 

### 5. The car is able to change lanes

The car is able to smoothly change lanes when it makes sense to do so, such as when behind a slower moving car and an adjacent lane is clear of other traffic. For the purpose of changing lanes, one variable indicating the goal lane index `lane` and two lane change variables `turn_left` and `turn_right` (code line 60, 123 and 124) are defined for the state change. The `lane` is defaulted at 1, which means the car drives normally on the center lane. As introduced in section 3, the lane change variable will be set according to the sensor fusion information, that is, when there is vehicle very near to the ego vehicle not allowing a lane change, then the variable should accordingly set to false (code lines 133 - 182). If the variable `lane` is changed then the planner would provide the spline interpolated trajectory points to the simulater to make the lane change happen.

### Rsult

This is a accelerated record of the path planner behavior for 4.32 miles.

![alt text][video1]
