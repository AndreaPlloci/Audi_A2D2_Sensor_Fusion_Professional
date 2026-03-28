"""
Entry point della pipeline di sensor fusion Audi A2D2.

Utilizzo:
    cd sensor-fusion
    python main.py

Configurare i percorsi in config.py prima dell'esecuzione.
"""

from config import A2D2_DATASET_ROOT, BUS_SIGNAL_DIR, OUTPUT_DIR
from ingestion.a2d2_loader import A2D2Loader
from ingestion.bus_parser import BusSignalParser
from fusion.fusion_engine import SensorFusionEngine
from fusion.scene_builder import SceneBuilder


def main():
    print("=== Audi A2D2 Sensor Fusion Pipeline ===")
    print()

    # ------------------------------------------------------------------
    # Configurazione sequenza da elaborare.
    # Modificare SEQUENCE_NAME con una sequenza disponibile localmente.
    # ------------------------------------------------------------------
    SEQUENCE_NAME = "20180807_145028"   # Esempio - sostituire con sequenza reale
    CAMERA_ID     = "cam_front_center"
    MAX_FRAMES    = 50                  # None per elaborare l'intera sequenza

    sequence_path = A2D2_DATASET_ROOT / "camera_lidar" / SEQUENCE_NAME

    # Inizializzazione loader
    print(f"Caricamento sequenza: {sequence_path}")
    loader = A2D2Loader(sequence_path)

    # Bus signals (opzionali - non tutte le sequenze li includono)
    bus_parser = None
    bus_signal_file = BUS_SIGNAL_DIR / f"{SEQUENCE_NAME}.json"
    if bus_signal_file.exists():
        print(f"Caricamento bus signals: {bus_signal_file}")
        bus_parser = BusSignalParser(bus_signal_file)
    else:
        print("Bus signals non disponibili per questa sequenza.")

    # Inizializzazione motore di fusione
    engine = SensorFusionEngine(loader, bus_parser, camera_id=CAMERA_ID)

    # Elaborazione sequenza
    label = f"max {MAX_FRAMES} frame" if MAX_FRAMES else "intera sequenza"
    print(f"\nElaborazione {label}...")
    fused_frames = engine.process_sequence(max_frames=MAX_FRAMES)
    print(f"\nElaborati {len(fused_frames)} frame con fusione completa.")

    # Costruzione scena 3D aggregata
    print("\nCostruzione scena 3D aggregata...")
    builder = SceneBuilder()
    scene = builder.build_from_sequence(fused_frames)

    print(f"  Punti totali nella scena: {len(scene.points):,}")
    bbox_min, bbox_max = scene.bounding_box
    print(f"  Bounding box min: {bbox_min}")
    print(f"  Bounding box max: {bbox_max}")

    # Salvataggio output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ply_path = OUTPUT_DIR / f"{SEQUENCE_NAME}_fused_scene.ply"
    builder.save_ply(scene, ply_path)

    print("\nPipeline completata con successo.")


if __name__ == "__main__":
    main()
