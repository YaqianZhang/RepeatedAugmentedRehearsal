

class recycle(object):

    def __init__(self):
        self.tmp_img=[]
        self.tmp_label=[]
    def store_tmp(self,img,label):
        self.tmp_img.append(img)
        self.tmp_label.append(label)
    def clear(self):
        self.tmp_img =[]
        self.tmp_img = []

    def retrieve_tmp(self):
        return self.tmp_img, self.tmp_label