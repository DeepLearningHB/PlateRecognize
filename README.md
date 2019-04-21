번호판 인식을 위한 모듈 사용 설명서
----
 1. 영상에서 번호판 영역을 추출하는 `PlateDetector` (아직 반영 안됨)
 2. 추출된 영상에서 번호판의 네 꼭지점의 위치를 찾는 `PlateAlign` (모델 제작 중)
 3. 번호판의 타입을 분류하는 `PlateType`
 4. 번호판에서 지역을 인식하는 `RegionRecognition`
 5. 번호판에서  숫자/문자를 인식하는 `CharRecognition`

 각 모듈 사용의 통일성을 위해 각 클래스는 `생성자`, `predict()`,  `predict_proba()` , `소멸자`로
 구성되어있고, 사용법은 모두 동일하다.

 ### 생성자
 각 모듈이 모두 동일하며, `ckpt`파일이 있는 경로를 인자로 받는다.
 매개변수로 받은 `ckpt`를 이용해 모델을 `load`하며 사용할 준비를 마친다.
 ```python
from platetype.platetype import PlateType

 typetest_path = './TypeTestImage/typetest.jpg'
 checkpoint_path_type = './platetype/model'
 a = PlateType(checkpoint_path_type)
 ```

 ### predict()
 모델의 추론 결과를 반환한다. 인자로는 Image가 들어간다. `predict()`와 `predict_proba()`에서 적절히 모델의 입력에 알맞게 이미지를 변형시켜 추론을 진행한다.
 ```python
print(a.predict(image))
 ```

 ### predict_proba()
 모델의 추론 결과를 `softmax`한 값으로 나타낸다.
 ```python
 print(a.predict_proba(image))
 ```
 ```
 [1.9521893e-08 9.9826658e-01 1.7299971e-03 1.8891556e-06 7.9598226e-07
 6.9683915e-07 8.8940062e-08 1.8145908e-08]
 ```

 ### 소멸자
 세션을 닫는다. 프로그램 종료 시 자동수행 되지만,
 `del`키워드를 이용한 삭제 및 `<Module_name>.close()`로도 수행 가능하다.

 #### 각 모듈의 라벨
 ###### PlateType
  - `0` : 번호판 없음
  - `1` : 일반적인 흰색 번호판
  - `2` : 위쪽에 나사 달린 흰색 번호판
  - `3` : 노란색 택시 번호판
  - `4` : 구형 녹색 번호판 (지역 x)
  - `5` : 지역이 표시된 녹색 번호판
  - `6` : 노란색 버스 번호판
  - `7` : 오토바이 번호판

###### CharRecognition
```
 ['0','1','2','3','4','5','6','7','8','9','아','바','배','버','보','부','다','더','도','두','어','가','거','고','구','하','허','호','자',
'저','조','주','라','러','로','루','마','머','모','무','나','너','노','누',
'오','사','서','소','수','우']
```

###### RegionRecognition
```
['부산', '충북','충남','대구','대전','강원','광주','경북','경기','경남','인천','제주','전북','전남','서울','울산','부산','충북','충남','대구','대전','강원','광주','경북','경기','경남','인천','제주','전북','전남','서울','울산','부산']
가로 문자열, 세로 문자열 분리
.
