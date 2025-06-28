from glob import glob

z=glob("../images/*/*png")+glob("../images/*/*jpg")
print(z)
