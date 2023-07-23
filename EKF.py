import cv2
import numpy as np
import time

class EKF(object):

	def __init__(self, ix, iy, vx, vy, q,dt):
		self.x = np.zeros(4)
		self.x[0] = ix
		self.x[1] = vx
		self.x[2] = iy
		self.x[3] = vy
		self.x=np.array([self.x]).T
		self.q=q
		self.dt=dt
		self.F=np.array([[1,self.dt,0,0],[0,1,0,0,],[0,0,1,self.dt],[0,0,0,1]])
		A=np.array([[((self.dt**3)/3),((self.dt**2)/2)],[((self.dt**2)/2),self.dt]])
		self.Q=np.array([[A[0][0],A[0][1],0,0],[A[1][0],A[1][1],0,0],[0,0,A[0][0],A[0][1]],[0,0,A[1][0],A[1][1]]])*q
		self.x_t1=self.x
		self.P = np.eye(4)
		self.P_t1 = self.P
		self.z,self.H=self.Jacobian(self.x)
		
		
	def Jacobian(self,x):
		r=np.sqrt(x[0][0]**2+x[2][0]**2)
		b=np.arctan2(x[2][0],x[0][0])*180/np.pi
		z=np.array([[r,b]]).T
		H=np.array([[np.cos(b),0,np.sin(b),0],[-1*np.sin(b)/r,0,np.cos(b)/r,0]])
		return(z,H)		
		
	def predict(self):
		self.x_t1= self.F.dot(self.x)
		self.P_t1=self.F.dot(self.P).dot(self.F.T)+self.Q
		return(self.x_t1[0][0],self.x_t1[2][0])
		
	def update(self,x1, y1, vx, vy, r):
		R=np.array([[r**2,0],[0,r**2]])
		x = np.zeros(4)
		x[0] = x1
		x[1] = vx
		x[2] = y1
		x[3] = vy
		x=np.array([x]).T
		z,H=self.Jacobian(x)
		y,_=self.Jacobian(self.x_t1)
		s=H.dot(self.P_t1).dot(H.T)+R
		K=self.P_t1.dot(H.T).dot(np.linalg.inv(s))
		self.x=self.x_t1+(K.dot(z-y))
		self.P=(np.eye(4)-K.dot(H)).dot(self.P_t1)

#t=EKF(0,1,2,3,4,1,1)
#t.predict()
#print(t.x)
#print(t.P)
#print(t.F)
#print(t.Q)
#print(t.x_t1)
#print(t.P_t1)
#print(t.z)
#print(t.H)
#t.update(1,3,2,4,3)
#print(t.x)
#print(t.P)


