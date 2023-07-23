import cv2
import numpy as np
import time
import EKF

def showimage(img):
	cv2.namedWindow('img', cv2.WINDOW_NORMAL)
	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# I declared a class BoundingBox to store the bounding box of each blob and to calculate center location.
class BoundingBox(object):

	def __init__(self, x, y, w, h):
		self.left = x
		self.right = x + w
		self.top = y
		self.bottom = y + h
		self.center_x = x + w / 2
		self.center_y = y + h / 2
		self.width = w
		self.height = h
		
	def area(self):
		return (self.right - self.left) * (self.bottom - self.top)

# I declared a class Blob to store the details of each blob- area, velocity, tracking id, age of blob, contour data, and parent data.
class Blob(object):

	def __init__(self, area, bounding_box, density, velocity, blob_index, frame_index, boxfit,contour,t):
		self.area = area
		self.bounding_box = bounding_box
		self.density = density
		self.__velocity = velocity
		self.blob_index = blob_index
		self.frame_index = frame_index
		self.__track_id = 0
		self.__age = 0
		self.boxfit = boxfit
		self.contour=contour
		self.t=t
		self.parents=[]
		self.d_parent=0
		self.kalman_filter=False
		self.px=0
		self.py=0
		self.p_px=0
		self.p_py=0
		self.p_bounding_box=False
		self.r=-1

	def get_new_EKF(self):
		self.kalman_filter=EKF.EKF(ix=self.bounding_box.center_x, iy=self.bounding_box.center_y, vx=self.__velocity[0], vy=self.__velocity[1], q=0.1,dt=1)
		return(self.kalman_filter)

	def EKF_predict(self):
		self.px,self.py=self.kalman_filter.predict()
		#print(self.px,self.py)
		return(self.px,self.py)

	def EKF_update(self,current_blobs_arr,current_blobs):
		if(self.kalman_filter==False):
			k=self.get_new_EKF()
			self.px,self.py=self.EKF_predict()
		else:
			if(self.r==-1):
				self.r= self.get_overlap_area(current_blobs_arr,current_blobs)
				#print(self.r)
				if(self.r<20):
					k=self.get_new_EKF()
					self.px,self.py=self.EKF_predict()
			self.kalman_filter.update(self.bounding_box.center_x, self.bounding_box.center_y, self.__velocity[0], self.__velocity[1], self.r)
			self.px,self.py=self.EKF_predict()
			
	def get_overlap_area(self,current_blobs_arr,current_blobs):
		r=0
		dx,dy=self.get_diff_position_p()
		for i in range(max(0,self.p_bounding_box.top-dy),min(len(current_blobs_arr),self.p_bounding_box.bottom-dy)):
			for j in range(max(0,self.p_bounding_box.left-dx),min(len(current_blobs_arr[0]),self.p_bounding_box.right-dx)):
				if(current_blobs_arr[i][j]!=0):
					r=r+1#current_blobs[current_blobs_arr[i][j]-1].density
		return(r)

	def get_diff_position(self,factor):
		#print([self.bounding_box.center_x,self.bounding_box.center_y],[int((self.bounding_box.center_x-self.px)),int((self.bounding_box.center_y-self.py))])
		return([int((self.bounding_box.center_x-self.px)*factor),int((self.bounding_box.center_y-self.py)*factor)])

	def get_diff_position_p(self):
		#print([self.bounding_box.center_x,self.bounding_box.center_y],[int((self.bounding_box.center_x-self.px)),int((self.bounding_box.center_y-self.py))])
		return([int((self.p_bounding_box.center_x-self.p_px)),int((self.p_bounding_box.center_y-self.p_py))])
		
	def get_age(self):
		return self.__age
	
	def set_age(self, age):
		self.__age = age

	def set_velocity(self, velocity):
		self.__velocity = velocity

	def get_velocity(self):
		return self.__velocity

	def set_track_id(self, track_id):
		self.__track_id = track_id

	def get_track_id(self):
		return self.__track_id

# I declared a class BlobsFrame to extract blobs from a given binary image and show the blobs on the raw image.
	
	def __str__(self):
		# return "blob_{}_left({}) top({}) width({}) height({})_v{}_frame_{}".format(
		# 	self.blob_index, self.bounding_box.left, self.bounding_box.top, self.bounding_box.width, self.bounding_box.height,
		# 	self.__velocity, self.frame_index)
		# 	
		return "blob id {} frame {} _ l({}) t({})".format(
			self.__track_id, self.frame_index, self.bounding_box.left, self.bounding_box.top)

# Is BlobsFrame init function to store binary image, raw image, and an array of blobs present in the current frame.
class BlobsFrame(object):

	def __init__(self, img, img_raw, frame_index, min_area=5):
		self.img = img
		self.img_out=img_raw.copy()
		self.img_out_bbox=img_raw.copy()
		self.frame_index = frame_index
		self.__min_area = min_area
		self.__blobs = []
		self.blob_arr=np.zeros_like(img, dtype = np.uint16)
		self.__extract_blobs()
	
# Is extract blob function. It uses OpenCV contours to detect blobs and to get the area, density, bounding box, and box points of each blob. 
# It returns an array of Blob class objects representing blobs and an image array with blob indexes as values 
# in the current frame which can be used as previous frame blobs details for the next frame.
	def __extract_blobs(self):
		
		img_processed = self.img
		frame = self.img_out
		contours, _ = cv2.findContours(img_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		i = 0
		j=-1
		for cnt in contours:
			j=j+1
			blob_area = cv2.contourArea(cnt)
			if (blob_area < self.__min_area):
				continue
			i += 1
			#rectangle fit
			rect = cv2.minAreaRect(cnt)
			box = cv2.boxPoints(rect)
			boxfit = np.int0(box)
			t=np.zeros_like(frame, dtype = np.uint16)
			cv2.drawContours(t, contours, j, (255), -1)
			t=np.asarray(t, dtype = np.uint16)
			self.blob_arr[t==255]=i
			x, y, w, h = cv2.boundingRect(cnt)
			bbox = BoundingBox(x, y, w, h)
			density = blob_area / bbox.area()
			blob = Blob(blob_area, bbox, density, [0, 0], i, self.frame_index, boxfit,cnt,t)
			self.__blobs.append(blob)
			cv2.drawContours(frame, contours, j, (255, 255, 255), -1)
		#print("min area = ", min(array_area))
		#print(i,self.blob_arr.max(),self.blob_arr.shape)


	def get_bbox_image(self, factor):
		self.img_out_bbox=self.img_out.copy()
		self.img_out_bbox = cv2.cvtColor(self.img_out_bbox, cv2.COLOR_GRAY2RGB)
		image_shape = self.img_out_bbox.shape
		height = image_shape[0] * factor
		width = image_shape[1] * factor

		# resize image
		self.img_out_bbox = cv2.resize(self.img_out_bbox, (width, height))
		for b in self.__blobs:
			x =  b.bounding_box.left * factor
			y =  b.bounding_box.top * factor
			w =  (b.bounding_box.right - b.bounding_box.left) * factor
			h =  (b.bounding_box.bottom - b.bounding_box.top) * factor
			boxfit = b.boxfit * factor

			cv2.drawContours(self.img_out_bbox,[boxfit],0, (255, 255, 0),1)
			cv2.putText(self.img_out_bbox, str(b.get_track_id()) + "age:" + str(b.get_age()), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))
		
		height = image_shape[0]
		width = image_shape[1]
		self.img_out_bbox = cv2.resize(self.img_out_bbox, (width, height))
		return(self.img_out_bbox)

	def get_blobs(self):
		return self.__blobs


class BlobVertex(object):

	def __init__(self, blob):

		self.blob = blob
		self.neighbors_blobs = {}
		self.__is_parent = False
		self.__is_descendent = False

	def __str__(self):
		return str(self.blob) + '--> ' + str([str(x.blob)for x in self.neighbors_blobs])

	def add_neighbor(self, blob, weight=0):
		self.neighbors_blobs[blob] = weight

	def remove_neighbor(self, blob):
		return self.neighbors_blobs.pop(blob, None)

	def get_neightbors(self):
		return self.neighbors_blobs.keys() 
	
	def has_neightbor(self, blob):
		return blob in self.neighbors_blobs.keys()

	def vertices_of_degree(self):
		return len(self.neighbors_blobs)
	
	def is_parent(self):
		return self.__is_parent
	
	def set_is_parent(self, v):
		self.__is_parent = v
	
	def is_descendent(self):
		return self.__is_descendent
	
	def set_is_descendent(self, v):
		self.__is_descendent = v

	def S_total_area_neighbors(self):
		S = sum([nb.blob.area for nb in self.get_neightbors()])
		return S

# I declared a class Graph to calculate blob graph, cost value, velocity, track id, and to get output image for this frame.
class Graph(object):

	def __init__(self, img_raw, current_blobs,current_blobs_arr,prior_blobs,prior_blobs_arr,max_track_id,track_id):
		self.img_raw=img_raw
		self.current_blobs=current_blobs
		self.current_blobs_arr=current_blobs_arr
		self.prior_blobs=prior_blobs
		self.prior_blobs_arr=prior_blobs_arr
		self.max_track_id=max_track_id
		self.track_id=track_id
		self.G_V_E=[]
		self.G_E_V=[]
		self.new=[]
		self.delete=[]
		self.split=[]
		self.merge=[]
		self.one_one_prior=[]
		self.one_one_current=[]
		self.img_out_bbox=[]

	def __add_vertex(self, blob_node):
		"""
			@parameters
			blob_node: (Blob) data key of BlobVertex node
		"""
		self.num_vertices = self.num_vertices + 1
		new_vertex = BlobVertex(blob_node)
		self.vert_dict[blob_node] = new_vertex
		return new_vertex

	def get_vertex(self, n):
		"""
			@parameters
			n: (Blob) get BlobVertex from data key
			return BlobVertex
		"""
		if n in self.vert_dict:
			return self.vert_dict[n]
		else:
			return None


	def __add_edge(self, frm, to, cost = 0):
		"""
			@parameters
			frm: (Blob) data from
			to: (Blob) data to
		"""
		if frm not in self.vert_dict:
			self.__add_vertex(frm)
		if to not in self.vert_dict:
			self.__add_vertex(to)

		# print("add edge: ", frm, "-> ", to)
		self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
		self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

	def __remove_edge(self, frm, to):
		"""
			@parameters
			frm: (Blob) data from
			to: (Blob) data to
		"""
		# print("do remove, ", frm, to)
		self.vert_dict[frm].remove_neighbor(self.vert_dict[to])
		self.vert_dict[to].remove_neighbor(self.vert_dict[frm])

	def __get_vertices(self):
		return self.vert_dict.keys()

	def __init_base_graph(self):
		"""
			blobs_prior: List (Blob) data frame i - 1
			blobs_current: List (Blob) data frame i
		"""
		for u in self.blobs_current:
			self.__add_vertex(u)
			for v in self.blobs_prior:
				if (self.__locality_constraint(u, v)):
					self.__add_edge(u, v)

	def __is_parent_vertex(self, vert):
		is_parent = False
		vert_degree = vert.vertices_of_degree()
		if (vert_degree > 1 or vert_degree == 0):
			is_parent = True
		if (vert_degree == 0 and vert.is_current_frame):
			is_parent = False
		return is_parent

	def __init_parents_and_descendents_set(self):
		
		for vert in self.__get_vertices():
			vert_degree = vert.vertices_of_degree()
			if (vert_degree > 1 or vert_degree == 0):
				vert.set_is_parent(True)
			if (vert_degree == 0 and vert.is_current_frame):
				vert.set_is_parent(True)
		
		# for vert in self.__get_vertices():
		# 	if (vert.is_parent()):
		# 		for nb in vert.get_neightbors():
		# 			nb.set_is_descendent(True)

		

	def caluculate_velocity_one_one(self):
		blobs=self.one_one_current
		beta=0.5
		#constant feame rate
		dt=1

		for i in range(len(blobs)):
			current_blob=self.current_blobs[blobs[i][0]-1]
			prior_blob=self.prior_blobs[blobs[i][1]-1]
			if(len(self.G_V_E[blobs[i][1]-1])>1):
				current_blob.set_velocity(prior_blob.get_velocity())

			else:
				x_velocity = beta * (current_blob.bounding_box.center_x - prior_blob.bounding_box.center_x) / dt + (1 - beta) * prior_blob.get_velocity()[0]
				y_velocity = beta * (current_blob.bounding_box.center_y - prior_blob.bounding_box.center_y) / dt + (1 - beta) * prior_blob.get_velocity()[1]
				current_blob.set_velocity([x_velocity, y_velocity])


	def caluculate_velocity_merge(self):
		blobs=self.merge
		for i in blobs:
			cureent_blob=self.current_blobs[i-1]
			max_area=0
			for j in range(len(self.G_E_V[i-1])):
				prior_blob=self.prior_blobs[self.G_E_V[i-1][j]-1]
				if(prior_blob.area>max_area):
					max_area=prior_blob.area
					cureent_blob.set_velocity(prior_blob.get_velocity())


	def cost(self):
		c=0.0
		for i in range(len(self.G_E_V)):
			cureent_blob=self.current_blobs[i]
			a_u=cureent_blob.area
			s_u=0
			for j in range (len(self.G_E_V[i])):
				prior_blob=self.prior_blobs[self.G_E_V[i][j]-1]
				s_u=s_u+prior_blob.area
			t=abs(a_u-s_u)/max(a_u,s_u)
			c=c+t
		return(c)


	def __locality_constraint(self, vertex_u, vertex_v):
		"""
			@parameters
			vertex_u: (Blob) key vertext u
			vertex_v: (Blob) key vertext v
		"""
		overlap_area = vertex_u.area_overlap_to(vertex_v)
		min_bbx_blob = min(vertex_u.bounding_box.area(), vertex_v.bounding_box.area())
		#IN PAPER IMPLEMENT min bb / 2
		is_valid = overlap_area >= min_bbx_blob / 2

		return is_valid

	def __parent_structure_constraint(self, blob_u, blob_v):
		"""
			@parameters
			blob_u: (Blob) key vertext u
			blob_v: (Blob) key vertext v
		"""
		is_valid = True
		list_violate_blobs = []
		vertex_u = self.get_vertex(blob_u)
		vertex_v = self.get_vertex(blob_v)
		if (vertex_u is None or vertex_v is None):
			is_valid = True
		elif (vertex_u.vertices_of_degree() > 1 or vertex_v.vertices_of_degree() > 1):
			for nb in vertex_u.get_neightbors():
				if vertex_v.has_neightbor(nb):
					is_valid = False
					# break
					list_violate_blobs.append([blob_u, nb.blob, blob_v])
		else:
			is_valid = True

		return is_valid, list_violate_blobs


	def optimize(self):

		for b in self.current_blobs:	
			t=np.copy(b.t)
			t[t==255]=self.prior_blobs_arr[t==255]
			t1=np.unique(t)
			if(t1[0]==0):
				t1=t1[1:]
			self.G_E_V.append(t1)
			if(len(t1)==0):
				self.new.append(b.blob_index)
			elif(len(t1)==1):
				self.one_one_current.append([b.blob_index,t1[0]])
			else:
				self.merge.append(b.blob_index)


		for b in self.prior_blobs:
			t=np.copy(b.t)			
			t[t==255]=self.current_blobs_arr[t==255]
			t1=np.unique(t)
			if(t1[0]==0):
				t1=t1[1:]
			self.G_V_E.append(t1)
			if(len(t1)==0):
				self.delete.append(b.blob_index)
			elif(len(t1)==1):
				self.one_one_prior.append([b.blob_index,t1[0]])
			else:
				self.split.append(b.blob_index)

		for i in range(len(self.delete)):
			prior_blob=self.prior_blobs[self.delete[i]-1]
			#print(prior_blob.blob_index,self.delete[i])
			self.track_id.append(prior_blob.get_track_id())
		#print(self.track_id)

		for i in range(len(self.new)):
			current_blob=self.current_blobs[self.new[i]-1]
			#print(current_blob.blob_index,self.new[i])
			try:
				t=self.track_id.pop(0)
			except:
				self.max_track_id=self.max_track_id+1
				t=self.max_track_id
			current_blob.set_track_id(t)
			current_blob.parents=[t]
			current_blob.set_age(0)

		for i in range(len(self.one_one_current)):
			b_id=self.one_one_current[i][0]
			b1_id=self.one_one_current[i][1]
			if(self.G_E_V[b_id-1][0]==b1_id)and(len(self.G_E_V[b_id-1])==1)and(len(self.G_V_E[b1_id-1])==1)and(self.G_V_E[b1_id-1][0]==b_id):
				#print(b_id,self.G_E_V[b_id-1],self.G_V_E[b1_id-1],b1_id)
				self.current_blobs[b_id-1].set_track_id(self.prior_blobs[b1_id-1].get_track_id())
				self.current_blobs[b_id-1].set_age(self.prior_blobs[b1_id-1].get_age()+1)
				self.current_blobs[b_id-1].parents=self.prior_blobs[b1_id-1].parents
				self.current_blobs[b_id-1].kalman_filter=self.prior_blobs[b1_id-1].kalman_filter
				self.current_blobs[b_id-1].p_px=self.prior_blobs[b1_id-1].px
				self.current_blobs[b_id-1].p_py=self.prior_blobs[b1_id-1].py
				self.current_blobs[b_id-1].p_bounding_box=self.prior_blobs[b1_id-1].bounding_box


		for i in range(len(self.split)):
			b_id=self.split[i]
			for j in range(len(self.G_V_E[b_id-1])):
				current_blob=self.current_blobs[self.G_V_E[b_id-1][j]-1]
				if(len(self.prior_blobs[b_id-1].parents)==0):
					try:
						t=self.track_id.pop(0)
					except:
						self.max_track_id=self.max_track_id+1
					t=self.max_track_id
					current_blob.set_track_id(t)
					current_blob.parents=[t]
					current_blob.set_age(0)
					if(j==len(self.G_V_E[b_id-1])-1)and(self.prior_blobs[b_id-1].d_parent!=0):
						self.track_id.append(self.prior_blobs[b_id-1].d_parent)
				#	print("0")

				elif(j==len(self.G_V_E[b_id-1])-1):
					if(len(self.prior_blobs[b_id-1].parents)>1):
						t=self.prior_blobs[b_id-1].d_parent
						current_blob.d_parent=t
						current_blob.parents=self.prior_blobs[b_id-1].parents
					else:
						t=self.prior_blobs[b_id-1].parents.pop(0)
						current_blob.parents=[t]
						if(self.prior_blobs[b_id-1].d_parent!=0):
							self.track_id.append(self.prior_blobs[b_id-1].d_parent)
					current_blob.set_track_id(t)
					current_blob.set_age(0)
				#	print("1")

				else:
					t=self.prior_blobs[b_id-1].parents.pop(0)
					current_blob.set_track_id(t)
					current_blob.parents=[t]
					current_blob.set_age(0)
				#	print("1")
				#print(current_blob.parents,current_blob.get_track_id(),self.prior_blobs[b_id-1].parents)


		for i in range(len(self.merge)):
			b_id=self.merge[i]
			current_blob=self.current_blobs[b_id-1]
			d_id=0
			ids=[]
			max_area=0
			t=1
			p=self.prior_blobs[self.G_E_V[b_id-1][0]-1].parents
			if(len(p)==0):
				p=current_blob.parents
			#print(p,self.prior_blobs[self.G_E_V[b_id-1][0]-1].get_track_id(),current_blob.parents)


			for j in range(len(self.G_E_V[b_id-1])):
				prior_blob=self.prior_blobs[self.G_E_V[b_id-1][j]-1]
				p1=prior_blob.parents
				if(len(p1)==[]):
					p1=current_blob.parents
				if(len(p1)==len(p)):
					for k in range(len(p1)):
						if(p1[k]==p[k]):
							t=0
				if(prior_blob.d_parent!=0):
					ids.append(prior_blob.d_parent)
	
			if(t==0):
				#print(p)
				if(len(p)==1):
					t=p[0]
				else:
					if(len(ids)>=1):
						t=ids.pop(0)
					else:
						try:
							t=self.track_id.pop(0)
						except:
							self.max_track_id=self.max_track_id+1
							t=self.max_track_id
					current_blob.d_parent=t
				current_blob.parents=p
				current_blob.set_track_id(t)
				current_blob.set_age(0)
				#print("0",current_blob.get_track_id(),current_blob.d_parent,current_blob.parents)
				
			else:
				ids=[]
				for j in range(len(self.G_E_V[b_id-1])):
					prior_blob=self.prior_blobs[self.G_E_V[b_id-1][j]-1]
					for k in prior_blob.parents:
						if not(k in current_blob.parents):
							current_blob.parents.append(k)
					current_blob.parents.sort()
					if(prior_blob.area>max_area)and(prior_blob.d_parent!=0):
						if(d_id!=0):
							ids.append(d_id)
						d_id=prior_blob.d_parent
						max_area=prior_blob.area
					elif(prior_blob.d_parent!=0):
						ids.append(prior_blob.d_parent)
				if(d_id==0):
					try:
						d_id=self.track_id.pop(0)
					except:
						self.max_track_id=self.max_track_id+1
						d_id=self.max_track_id
				current_blob.set_track_id(d_id)
				current_blob.d_parent=d_id
				current_blob.set_age(0)
				#print("1",current_blob.get_track_id(),current_blob.d_parent,current_blob.parents)
			for j in ids:
				self.track_id.append(j)
						
		self.caluculate_velocity_one_one()
		self.caluculate_velocity_merge()
		for b in self.current_blobs:
			b.EKF_update(self.current_blobs_arr,self.current_blobs)
					
			
			
	def get_bbox_image(self, factor):
		self.img_out_bbox=self.img_raw.copy()
		self.img_out_bbox = cv2.cvtColor(self.img_out_bbox, cv2.COLOR_GRAY2RGB)
		image_shape = self.img_out_bbox.shape
		height = image_shape[0] * factor
		width = image_shape[1] * factor

		# resize image
		self.img_out_bbox = cv2.resize(self.img_out_bbox, (width, height))
		for b in self.current_blobs:
			x =  b.bounding_box.left * factor
			y =  b.bounding_box.top * factor
			w =  (b.bounding_box.right - b.bounding_box.left) * factor
			h =  (b.bounding_box.bottom - b.bounding_box.top) * factor
			boxfit = b.boxfit * factor
			boxfit1=boxfit+(b.get_diff_position(0.5))
			#print(boxfit,boxfit1)

			cv2.drawContours(self.img_out_bbox,[boxfit],0, (225, 0, 0),3)
			try:
				cv2.drawContours(self.img_out_bbox,[boxfit1],0, (0, 255, 0),3)
			except:
				print(boxfit,boxfit1)
			cv2.putText(self.img_out_bbox, str(b.get_track_id()) + "age:" + str(b.get_age()), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))
		#height = image_shape[0]
		#width = image_shape[1]
		#self.img_out_bbox = cv2.resize(self.img_out_bbox, (width, height))
		return(self.img_out_bbox)

