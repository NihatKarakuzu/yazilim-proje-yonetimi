# Yazılım Proje Yönetimi

Bu depo, dönem projesi kapsamında geliştirilecek görüntü sahteciliği tespit sistemi için hazırlanmıştır.
Amaç, klasik görüntü işleme yöntemleri ve yapay zeka yaklaşımları ile bir görselin değiştirilip değiştirilmediğini analiz etmektir.

## Proje Özeti

Proje iki ana analiz katmanından oluşur:

- **Klasik Yöntemler:** SIFT, SURF, AKAZE, ORB
- **Yapay Zeka Yöntemleri:** CNN tabanlı modeller ve ek AI yaklaşımı

Sistem, yüklenen görseller için karşılaştırma/tespit sonuçlarını üretir, raporlanabilir bir çıktıya dönüştürür ve sunuma uygun biçimde gösterir.

## Temel Hedefler

- Görsel yükleme ve analiz akışını uçtan uca çalıştırmak
- Klasik algoritmalarla özellik eşleştirme tabanlı sahtecilik tespiti yapmak
- AI modelleri ile sahtecilik olasılığı üretmek
- Sonuçları anlaşılır bir arayüzde sunmak
- Projeyi ekip çalışmasına uygun, versiyon kontrollü ve düzenli bir yapıda yönetmek

## Teknoloji Planı

- **Backend:** Python + FastAPI
- **Görüntü İşleme:** OpenCV
- **Yapay Zeka:** PyTorch (veya TensorFlow)
- **Veritabanı (opsiyonel ama önerilir):** PostgreSQL
- **Sürüm Kontrol:** Git + GitHub

## Ekip

- Nihat Karakuzu (`@NihatKarakuzu`)
- Sefa Özkan (`@ozkn-sefa`)
- Mert Evran (`@Mert-exe`)

## Geliştirme Aşamaları

1. Proje iskeleti, ortam kurulumu ve temel API
2. Klasik algoritmaların entegrasyonu (ORB -> AKAZE -> SIFT/SURF)
3. AI model eğitimi ve inference hattı
4. Sonuçların kayıt altına alınması ve raporlama
5. Arayüz iyileştirmeleri ve teslim dokümantasyonu

## Git Çalışma Düzeni

- Ana dal: `main`
- Geliştirme dalları:
  - `feature/backend-api`
  - `feature/opencv-classical`
  - `feature/ai-model`
  - `feature/frontend-ui`
  - `docs/user-manual`

Her geliştirme adımı küçük ve anlamlı commitlerle ilerletilir; düzenli push ve gerektiğinde PR akışı kullanılır.

## İlk Çalışan Sürüm (Backend API)

Bu aşamada temel backend API altyapısı kurulmuştur:

- `GET /api/health` -> servis durumunu döner
- `POST /api/upload` -> desteklenen görsel dosyasını yükler

### Çalıştırma

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Varsayılan adres: `http://127.0.0.1:8000`  
Dokümantasyon: `http://127.0.0.1:8000/docs`
Kullanıcı arayüzü: `http://127.0.0.1:8000/`

### PostgreSQL Bağlantısı (Opsiyonel ama önerilir)

Analiz sonuçlarının veritabanına kaydedilmesi için `DATABASE_URL` değişkeni tanımlanır.

Windows PowerShell örneği:

```powershell
$env:DATABASE_URL="postgresql://postgres:PAROLA@localhost:5432/yazilim_proje"
py -m uvicorn app.main:app --reload
```

`DATABASE_URL` yoksa API çalışmaya devam eder, sadece veritabanı kaydı yapılmaz.

## Klasik Analiz (2. Aşama)

Bu aşamada klasik algoritmalar genişletildi:

- `POST /api/analyze/orb`
  - `reference_image`: referans görsel
  - `test_image`: analiz edilecek görsel

- `POST /api/analyze/classical`
  - ORB + AKAZE + SIFT sonuçlarını tek yanıtta döner
  - her algoritma için karar (`authentic_like` veya `suspicious`) ve metrik üretir

## AI Analiz (3. Aşama - İlk Prototip)

Bu aşamada AI katmanı için ilk çalışan prototip endpointi eklendi:

- `POST /api/analyze/ai`
  - tek görsel alır ve iki farklı model yaklaşımından skor üretir
  - model çıktılarının ortalaması ile birleşik karar döner

Not: Bu sürüm, gerçek model eğitimi öncesi çalışan prototip akıştır. Sonraki adımda eğitimli CNN/LSTM modelleri entegre edilecektir.

## Sonuç Geçmişi

- `GET /api/analysis/history?limit=20`
  - Veritabanına kaydedilmiş son analiz sonuçlarını listeler.
