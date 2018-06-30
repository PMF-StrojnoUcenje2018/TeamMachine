# TeamMachine

Ovo je repozitorij napravljen za projekt iz kolegija Strojno učenje s temom [Google Landmark Retrieval Challenge](https://www.kaggle.com/c/landmark-retrieval-challenge/).

Zadatak je preuzet s Kaggle-a.

Zahtjevi za sustav su sljedeći:
- python
- keras
- cv2
- faiss

Također potrebno je postaviti `sample_submission.csv` u `input/` folder i kreirati `output/` folder.

Naše rješenje se pokreće na dva načina ovisno o modelu na kojem se želi pretrenirati mreža.

### VGG16

Dohvatimo prvo [slike](https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/56194) koje je već pripremio jedan korisnik u dimenzijama 128x128.
Raspakiramo dohvaćenu datoteke te u `input/` postavimo foldere `train/` i `test/`.

Zatim redom pokrećemo skripte
- `VGG16FeatExtractor.py` za feature extraction
- `output_submission.py` za dobivanje najsličnijih slika računajući Euklidsku udaljenost između vektora značajki
- alternativno može se pozvati `output_submission_v2.py`za računanje pomoću kosinusove sličnosti
 
Rješenje se nalazi u folderu `output/` u `sub_vgg_pred.csv`. U njemu je uz svaki id test slike nalazi 100 najsličnijih predikcija.

### ResNet50

Dohvatimo prvo .csv filove sa [stranice](https://www.kaggle.com/google/google-landmarks-dataset).
input/test.csv input/query
Raspakiramo dohvaćenu datoteke te u `input/` postavimo `index.csv` i `test.csv`, zatim kreiramo foldere `index/` i `query/`. 

Zatim redom pokrećemo skripte
- `imagesDownloader.py` kojoj kao argumente komandne linije prvo šaljemo `input/index.csv input/index`, te zatim ponovno sa argumentima `input/query.csv input/query`. Ova skripta dohvaća slike u RGB slike dimenzija 256x256 u .jpg formatu.
- `ResNetFeatExtractor.py` za feature extraction
- `output_submission_ResNet.py` za dobivanje najsličnijih slika računajući Euklidsku udaljenost između vektora značajki
 
Rješenje se nalazi u folderu `output/` u `sub_resnet50_pred.csv`. U njemu je uz svaki id test slike nalazi 100 najsličnijih predikcija.

##### Evaluacija rješenja
Evaluacija rješenja se provodi uploadom pripadajućeg .csv file-a na Kaggle challenge.
