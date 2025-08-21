param(
    [Parameter(Mandatory=$true)][string]$Repo,          # <owner/repo>
    [string]$Homepage = "https://www.linkedin.com/in/faik-döner"
)

# Bu script, GitHub gh CLI gerektirir: https://cli.github.com/
# Örnek kullanım:
#   .\scripts\setup_repo.ps1 -Repo "kullanici-adi/datascope" -Homepage "https://ornek.site"

$DescriptionTR = "DataScope: Streamlit tabanlı veri analizi. CSV/XLSX yükle, özellik seç, kümeleme ve sınıflandırmayı görselleştir."
$DescriptionEN = "DataScope: Streamlit-based data analysis. Upload CSV/XLSX, select features, run clustering/classification."
$Description = $DescriptionTR + " | " + $DescriptionEN

# Açıklama ve anasayfa
gh repo edit $Repo --description "$Description" --homepage "$Homepage"

# Önerilen konular
$topics = @(
  "streamlit","data-analysis","machine-learning","python",
  "clustering","classification","kmeans","dbscan","hierarchical-clustering",
  "random-forest","svm","xgboost","lightgbm","feature-engineering","roc-curve","confusion-matrix"
)

foreach ($t in $topics) {
  gh repo edit $Repo --add-topic $t | Out-Null
}

Write-Host "✓ Repository About bilgileri güncellendi: $Repo" -ForegroundColor Green
