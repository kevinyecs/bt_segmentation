# Agytumor szegmentáció konvolúciós hálózatok használatával MRI képeken 
A szakdolgozatom által felhasznált kódbázist, tartalmazza az alábbi git repo.
Futtatáshoz, szükséges még az adatbázis letöltése is a shared mappába.
A tanítások Paperspace oldalán lettek megvalósítva 16GB-os távoli elérésű videokártyákokal.

## Tanítás lépései
-thesis.ipynb futtatása
-fájlon belűl a Configurator osztályal tudunk kísérleteket inicializálni pl.: Architektúra, epochszám, batch méret, stb..
-Ha tanítást logolni szeretnénk megkell adni egy Weights And Biases kulcsot a felhasználónak.
-Vizualizálások, előfeldolgozás, augmentáció, tanítás újra inditása automatikusan történik

## Eredmények

A betanított hálózatok jó eredményekkel megtudták oldani a felvettett problémát és más fejlett algoritmusokkal
azonos vagy jobb eredményt volt képes elérni. 

## Felhasznált eszközök
Első sorban a tanítások Paperspace által nyújtott távolíelérésű gépeken voltak megvalósítva, csak a szignifikáns kódrészletek kerültek publikálásra ebben a repoban.
További felhasznált könyvtárak és keretrendszerek:
-Pytorch
-MONAI
-Matplotlib
-Weights And Biases
