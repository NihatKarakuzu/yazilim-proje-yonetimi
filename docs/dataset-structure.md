# Dataset Yapısı

Eğitim scripti, `torchvision.datasets.ImageFolder` yapısını bekler.

## Klasör Düzeni

```text
dataset/
  train/
    authentic/
      img1.jpg
      img2.jpg
      ...
    fake/
      img1.jpg
      img2.jpg
      ...
```

- `authentic` sınıf etiketi: `0`
- `fake` sınıf etiketi: `1`

## Notlar

- Desteklenen formatlar: jpg, jpeg, png, bmp, webp
- Sınıf sayıları olabildiğince dengeli olmalıdır
- Eğitim öncesi bozuk/okunamayan dosyaları temizleyin
