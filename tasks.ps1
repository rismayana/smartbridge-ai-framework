# SHMS Bridge Anomaly Detection - Pipeline Automation (MLflow + DagsHub)
# Cara pakai di PowerShell:
#   .\tasks.ps1 help
#   .\tasks.ps1 install
#   .\tasks.ps1 train
#
# Jika ada error "running scripts is disabled":
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

param(
    [Parameter(Position=0)]
    [string]$Task = "help"
)

$CODE     = "03_code"
$PIPELINE = "$CODE\run_pipeline.py"
$CFG      = "$CODE\config_loader.py"
$MLFLOW   = "$CODE\mlflow_config.py"

function Show-Help {
    Write-Host ""
    Write-Host "  SHMS Anomaly Detection - Pipeline Tasks (MLflow + DagsHub)" -ForegroundColor Cyan
    Write-Host "  ============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  SETUP" -ForegroundColor Yellow
    Write-Host "    install        Install semua dependencies (requirements.txt)"
    Write-Host "    check-config   Verifikasi config.yaml terbaca dengan benar"
    Write-Host "    check-mlflow   Cek koneksi MLflow ke DagsHub"
    Write-Host ""
    Write-Host "  DATA (DVC)" -ForegroundColor Yellow
    Write-Host "    data-pull      Pull data processed dari DagsHub remote"
    Write-Host "    data-push      Push data processed ke DagsHub remote"
    Write-Host ""
    Write-Host "  PIPELINE" -ForegroundColor Yellow
    Write-Host "    preprocess     Phase 1 + 2 (EDA + preprocessing)"
    Write-Host "    batch          Batch processor (data raw dari HD portable)"
    Write-Host "    train          Phase 3B - Isolation Forest (lokal, CPU)"
    Write-Host "    retrain        Phase 3B - paksa training ulang"
    Write-Host "    ensemble       Phase 4 - Ensemble fusion"
    Write-Host "    evaluate       Phase 5 - Evaluasi dan ablation study"
    Write-Host "    pipeline       Phase 3B + 4 + 5 (pipeline lokal penuh)"
    Write-Host ""
    Write-Host "  MLFLOW" -ForegroundColor Yellow
    Write-Host "    mlflow-ui         Buka MLflow UI lokal (http://localhost:5000)"
    Write-Host ""
    Write-Host "  MODEL REGISTRY" -ForegroundColor Yellow
    Write-Host "    registry-list     List semua model terdaftar + stage"
    Write-Host "    registry-register Daftarkan IForest terbaik ke Staging"
    Write-Host "    registry-promote  Promosikan model ke Production"
    Write-Host ""
    Write-Host "  Untuk LSTM dan GNN: jalankan notebook di Google Colab" -ForegroundColor DarkGray
    Write-Host "    03_code/notebooks/SHMS_Phase3A_LSTM_Autoencoder.ipynb" -ForegroundColor DarkGray
    Write-Host "    03_code/notebooks/SHMS_Phase3C_GNN.ipynb" -ForegroundColor DarkGray
    Write-Host ""
}

switch ($Task) {

    # Setup
    "install" {
        Write-Host "[install] pip install -r requirements.txt" -ForegroundColor Cyan
        python -m pip install -r requirements.txt
    }
    "check-config" {
        Write-Host "[check-config] Verifikasi config.yaml" -ForegroundColor Cyan
        python $CFG
    }
    "check-mlflow" {
        Write-Host "[check-mlflow] Cek koneksi MLflow ke DagsHub" -ForegroundColor Cyan
        python $MLFLOW
    }

    # DVC
    "data-pull" {
        Write-Host "[data-pull] dvc pull dari DagsHub" -ForegroundColor Cyan
        python -m dvc pull
    }
    "data-push" {
        Write-Host "[data-push] dvc push ke DagsHub" -ForegroundColor Cyan
        python -m dvc push
    }

    # Pipeline
    "batch" {
        Write-Host "[batch] Batch processor - data raw" -ForegroundColor Cyan
        python $PIPELINE --phase batch
    }
    "preprocess" {
        Write-Host "[preprocess] Phase 1 + 2" -ForegroundColor Cyan
        python $PIPELINE --phase 1 2
    }
    "train" {
        Write-Host "[train] Phase 3B - Isolation Forest" -ForegroundColor Cyan
        python $PIPELINE --phase 3b
    }
    "retrain" {
        Write-Host "[retrain] Phase 3B - paksa training ulang" -ForegroundColor Cyan
        python $PIPELINE --phase 3b --retrain
    }
    "ensemble" {
        Write-Host "[ensemble] Phase 4 - Ensemble fusion" -ForegroundColor Cyan
        python $PIPELINE --phase 4
    }
    "evaluate" {
        Write-Host "[evaluate] Phase 5 - Evaluasi akhir" -ForegroundColor Cyan
        python $PIPELINE --phase 5
    }
    "pipeline" {
        Write-Host "[pipeline] Phase 3B + 4 + 5" -ForegroundColor Cyan
        python $PIPELINE --phase 3b 4 5
    }

    # MLflow
    "mlflow-ui" {
        Write-Host "[mlflow-ui] Membuka MLflow UI di http://localhost:5000" -ForegroundColor Cyan
        Write-Host "Tekan Ctrl+C untuk berhenti." -ForegroundColor DarkGray
        python -m mlflow ui --backend-store-uri mlruns/ --port 5000
    }

    # Model Registry
    "registry-list" {
        Write-Host "[registry-list] List semua model di registry" -ForegroundColor Cyan
        python 03_code/model_registry.py list
    }
    "registry-register" {
        Write-Host "[registry-register] Daftarkan model IForest terbaik ke Staging" -ForegroundColor Cyan
        python 03_code/model_registry.py register --stage Staging
    }
    "registry-promote" {
        Write-Host "[registry-promote] Promosikan model ke stage tertentu" -ForegroundColor Cyan
        Write-Host "Contoh: python 03_code/model_registry.py promote iforest 1 --stage Production" -ForegroundColor DarkGray
        python 03_code/model_registry.py promote iforest 1 --stage Production
    }

    # Default
    default {
        if ($Task -ne "help") {
            Write-Host "Task '$Task' tidak dikenal." -ForegroundColor Red
        }
        Show-Help
    }
}
