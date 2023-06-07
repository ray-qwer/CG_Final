
dragon_cat = {
    "img_path":"drawing_data/dragon_cat.jpg",
    "skeleton_path":"drawing_data/dragon_cat_skeleton.npy",
    "segmask_config":{"D1_iter":0, "D2_iter":0},
}

bear = {
    "img_path":"drawing_data/bear.jpg",
    "skeleton_path":"drawing_data/bear_skeleton.npy",
    "segmask_config":{"D1_kernel":13, "D2_kernel":7, "D1_iter":2, "D2_iter":2, "blockSize":11, "tolerance":8},
}

maoli = {
    "img_path":"drawing_data/maoli.jpg",
    "skeleton_path":"drawing_data/maoli_skeleton.npy",
    "segmask_config":{"D1_kernel":13, "D2_kernel":3, "D1_iter":3, "D2_iter":0, "blockSize":15, "tolerance":1},
}

shit = {
    "img_path":"drawing_data/shit.jpg",
    "skeleton_path":"drawing_data/shit_skeleton.npy",
    "segmask_config":{"D1_kernel":9, "D1_iter":2, "D2_iter":0, "blockSize":15, "tolerance":1},
}

stickman = {
    "img_path":"drawing_data/stickman.jpg",
    "skeleton_path":"drawing_data/stickman_skeleton.npy",
    "segmask_config":{"D1_kernel":7, "D1_iter":1, "D2_iter":0, "blockSize":15, "tolerance":5},
}

stickman1 = {
    "img_path":"drawing_data/stickman1.jpg",
    "skeleton_path":"drawing_data/stickman1_skeleton.npy",
    "segmask_config":{"D1_kernel":9, "D1_iter":1, "D2_iter":0, "blockSize":17, "tolerance":2},
}

ghost = {
    "img_path":"drawing_data/ghost.jpg",
    "skeleton_path":"drawing_data/ghost_skeleton.npy",
    "segmask_config":{"D1_kernel":11, "D1_iter":2, "D2_kernel":7, "D2_iter":1, "blockSize":49, "tolerance":2},
}

pig = {
	"img_path":"drawing_data/pig.jpg",
	"skeleton_path":"drawing_data/pig_skeleton.npy",
	"segmask_config":{"D1_kernel":3, "D1_iter":2, "D2_kernel":3, "D2_iter":1, "blockSize":25, "tolerance":2},
}

pig2 = {
	"img_path":"drawing_data/pig2.jpg",
	"skeleton_path":"drawing_data/pig2_skeleton.npy",
	"segmask_config":{"D1_kernel":3, "D1_iter":2, "D2_kernel":3, "D2_iter":1, "blockSize":25, "tolerance":2},
}

dust = {
	"img_path":"drawing_data/dust.jpg",
	"skeleton_path":"drawing_data/dust_skeleton.npy",
	"segmask_config":{"D1_kernel":3, "D1_iter":2, "D2_kernel":3, "D2_iter":1, "blockSize":25, "tolerance":2},
}

fig_choices = ["dragon_cat", "bear", "maoli", "shit", "stickman","stickman1","ghost","pig","pig2","dust"]

def choose_drawing(name):	
	table = {"dragon_cat": dragon_cat,
	  		 "bear": bear,
			 "maoli": maoli,
			 "shit": shit,
			 "stickman": stickman,
			 "stickman1": stickman1,
			 "ghost": ghost,
	         "pig": pig,
	         "pig2": pig2,
	         "dust": dust}
	return table[name]