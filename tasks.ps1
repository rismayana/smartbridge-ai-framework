# SHMS Bridge Anomaly Detection - Pipeline Automation (W&B + Google Drive DVC)
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

function Show-Help {
    Write-Host ""
    Write-Host "  SHMS Anomaly Detection - Pipeline Tasks (W&B + Google Drive DVC)" -ForegroundColor Cyan
    Write-Host "  ==================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  SETUP" -ForegroundColor Yellow
    Write-Host "    install        Install semua dependencies (requirements.txt)"
    Write-Host "    check-config   Verifikasi config.yaml terbaca dengan benar"
    Write-Host "    check-wandb    Cek koneksi W&B"
    Write-Host ""
    Write-Host "  DATA (DVC)" -ForegroundColor Yellow
    Write-Host "    data-pull      Pull data processed dari Google Drive remote"
    Write-Host "    data-push      Push data processed ke Google Drive remote"
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
    Write-Host "  W&B" -ForegroundColor Yellow
    Write-Host "    wandb-sync     Sync run offline ke wandb.ai"
    Write-Host "    wandb-check    Cek status login W&B"
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
    "check-wandb" {
        Write-Host "[check-wandb] Cek koneksi W&B" -ForegroundColor Cyan
        python -c "import wandb; print('W&B version:', wandb.__version__); wandb.login()"
    }

    # DVC
    "data-pull" {
        Write-Host "[data-pull] dvc pull dari Google Drive" -ForegroundColor Cyan
        python -m dvc pull
    }
    "data-push" {
        Write-Host "[data-push] dvc push ke Google Drive" -ForegroundColor Cyan
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

    # W&B
    "wandb-sync" {
        Write-Host "[wandb-sync] Sync offline runs ke wandb.ai" -ForegroundColor Cyan
        Write-Host "Pastikan WANDB_API_KEY sudah di-set di .env" -ForegroundColor DarkGray
        python -m wandb sync wandb/
    }
    "wandb-check" {
        Write-Host "[wandb-check] Verifikasi login W&B" -ForegroundColor Cyan
        python -m wandb login --verify
    }

    # Default
    default {
        if ($Task -ne "help") {
            Write-Host "Task '$Task' tidak dikenal." -ForegroundColor Red
        }
        Show-Help
    }
}
