"""
Parser per i segnali bus del veicolo (CAN bus) nel dataset A2D2.
I bus signals includono velocità, accelerazione, angolo di sterzata e altri parametri dinamici.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BusSignalFrame:
    """Stato dinamico del veicolo in un dato istante temporale."""
    timestamp: int
    velocita_kmh: float             # Velocità longitudinale in km/h
    accelerazione_x: float          # Accelerazione longitudinale in m/s²
    accelerazione_y: float          # Accelerazione laterale in m/s²
    angolo_sterzo_deg: float        # Angolo volante in gradi
    velocita_angolare_yaw: float    # Velocità angolare yaw in rad/s
    angolo_pitch_deg: float         # Angolo pitch del veicolo in gradi
    angolo_roll_deg: float          # Angolo roll del veicolo in gradi


class BusSignalParser:
    """
    Carica e interroga i segnali bus del dataset A2D2.

    Il file JSON ha struttura:
        { "<timestamp_us>": { "<segnale>": { "value": ... } } }
    """

    def __init__(self, bus_signal_path: Path):
        if not bus_signal_path.exists():
            raise FileNotFoundError(f"File bus signals non trovato: {bus_signal_path}")

        with open(bus_signal_path, "r") as f:
            raw = json.load(f)

        self._frames: list[BusSignalFrame] = []
        self._timestamps: np.ndarray = np.array([], dtype=np.int64)
        self._parse(raw)

    def _parse(self, raw: dict) -> None:
        """Converte il JSON grezzo in lista ordinata di BusSignalFrame."""
        for ts_str, signals in raw.items():
            try:
                frame = BusSignalFrame(
                    timestamp=int(ts_str),
                    velocita_kmh=signals.get("vehicle speed", {}).get("value", 0.0),
                    accelerazione_x=signals.get("acceleration x", {}).get("value", 0.0),
                    accelerazione_y=signals.get("acceleration y", {}).get("value", 0.0),
                    angolo_sterzo_deg=signals.get("steering angle calculated", {}).get("value", 0.0),
                    velocita_angolare_yaw=signals.get("angular velocity omega", {}).get("value", 0.0),
                    angolo_pitch_deg=signals.get("pitch angle", {}).get("value", 0.0),
                    angolo_roll_deg=signals.get("roll angle", {}).get("value", 0.0),
                )
                self._frames.append(frame)
            except (KeyError, TypeError):
                continue

        self._frames.sort(key=lambda f: f.timestamp)
        self._timestamps = np.array([f.timestamp for f in self._frames], dtype=np.int64)

    def get_nearest(self, query_timestamp: int, tolerance_us: int = 50_000) -> BusSignalFrame | None:
        """
        Restituisce il BusSignalFrame più vicino al timestamp dato.
        Ritorna None se nessun frame è entro la tolleranza specificata.
        """
        if len(self._timestamps) == 0:
            return None

        idx = int(np.searchsorted(self._timestamps, query_timestamp))
        idx = max(0, min(idx, len(self._frames) - 1))

        candidate = self._frames[idx]
        if abs(candidate.timestamp - query_timestamp) <= tolerance_us:
            return candidate

        return None

    def get_all_frames(self) -> list[BusSignalFrame]:
        """Restituisce tutti i frame bus signal in ordine cronologico."""
        return self._frames
