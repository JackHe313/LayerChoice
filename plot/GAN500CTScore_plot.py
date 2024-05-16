#plot three graph horizontally
import matplotlib.pyplot as plt
import numpy as np

base = [-0.4541984574163905, 0.7327393297363205, 1.1574292804159465, 1.3571785973469483, 1.5258974129350564, 1.6960392909299065, 1.8921369814219948, 2.022168171852389, 2.4109196539685938, 2.4385116848364166, 3.2893466359798116, 2.596752702279013]
large = [-0.18645754807140433, 0.22446443925796153, 0.7690762786664808, 0.3820744990366485, 0.839237333090921, 0.9327303041967717, 0.9758202682714697, 1.4639870335825058, 1.4271115702973884, 2.1363992003373222, 2.7575905399235494, 2.6892745756649217, 2.9841681237428404, 3.2118909352911578, 3.8332756729914257, 3.898959668642179, 4.519189978069703, 4.906423386684316, 5.5936088935612895, 5.177961336824812, 5.775661844661665, 5.86048077142219, 6.211211816483271, 5.478716187960471]
huge = [-0.09209149604105499, 0.21861133479727496, 0.20992107012749936, 0.3488348808373233, 0.39367359772224153, 0.6895282294289073, 0.894964474616142, 1.078597677803062, 1.2895070414760414, 1.3745254715049176, 1.437271096672117, 1.54627167799116, 1.7465038659108922, 1.8225313176051139, 1.9732308531152407, 2.197762465964656, 2.4053871367386988, 2.3252537028187588, 2.6661905713684027, 2.7812639486780424, 2.8741628035168687, 3.1662163054956975, 3.251113803421962, 3.4314488068226585, 3.5479526397901635, 3.6393714635333896, 3.725332283839865, 3.9223054011006737, 3.8926120788404805, 3.6597445858160516, 3.8413461853344075, 3.565220438773622]

x1 = np.arange(1, len(base)+1)
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(x1, base, 'o-', label='base')
plt.ylabel("CT scores")
plt.xlabel("layers")
plt.title(f"Encoder ViT Base")
plt.legend()
plt.grid(True)
plt.subplot(132)
x2 = np.arange(1, len(large)+1)
plt.plot(x2, large, 'o-', label='large')
plt.xlabel("layers")
plt.title(f"Encoder ViT Large")
plt.legend()
plt.grid(True)
plt.subplot(133)
x3 = np.arange(1, len(huge)+1)
plt.plot(x3, huge, 'o-', label='huge')
plt.xlabel("layers")
plt.title(f"Encoder ViT Huge")
plt.legend()
plt.grid(True)
plt.show()