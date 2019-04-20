from platetype.platetype import PlateType
from charrecognition.charrecognition import CharRecognition
from regionrecognition.regionrecognition import RegionRecognition
from PIL import Image

checkpoint_path_type = '/home/leehanbeen/PycharmProjects/PlateRecognizeSystem/platetype/model'
checkpoint_path_char = '/home/leehanbeen/PycharmProjects/PlateRecognizeSystem/charrecognition/model'
checkpoint_path_region = '/home/leehanbeen/PycharmProjects/PlateRecognizeSystem/regionrecognition/model'
typetest_path = './TypeTestImage/typetest.jpg'
chartest_path = './TestCharImage/chartest.jpg'
regiontest_path = './RegionTestImage/testregion.jpg'
image = Image.open(typetest_path)
a = PlateType(checkpoint_path_type)
b = CharRecognition(checkpoint_path_char)
c = RegionRecognition(checkpoint_path_region)
print(a.predict(image))
print(a.predict_proba(image))
image = Image.open(chartest_path)
print(b.predict(image))
print(b.predict_proba(image))
image = Image.open(regiontest_path)
print(c.predict(image))
print(c.predict_proba(image))
