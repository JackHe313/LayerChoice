#plot three graph horizontally
import matplotlib.pyplot as plt
import numpy as np

small = [-1.0622082764933325, -0.41779658109856005, 0.0036793548438161794, 0.7482053637546526, 0.9104967961805341, 1.395090632469298, 1.6700572451340965, 1.8469200678959878, 1.7678316970116645, 2.369657072094719, 2.993291357572278, 3.2229617651188613]
base = [-0.9965388462414989, 0.12077901173264828, 0.9030798108213289, 1.3162229262265064, 1.5415794645139698, 1.667082779403247, 1.858848751638567, 1.7599231449220727, 2.299729733642124, 3.1596352099305394, 3.5065839029395196, 3.7009900252635854]

x1 = np.arange(1, len(base)+1)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(x1, base, 'o-', label='base')
plt.ylabel("CT scores")
plt.xlabel("layers")
plt.title(f"Encoder ViT Base")
plt.legend()
plt.grid(True)
plt.subplot(122)
x2 = np.arange(1, len(small)+1)
plt.plot(x2, small, 'o-', label='small')
plt.xlabel("layers")
plt.title(f"Encoder ViT Large")
plt.legend()
plt.grid(True)
plt.show()