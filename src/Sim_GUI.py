import sys
import os
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from config_schema import Config, Rules, Shoe
from PyQt5.QtGui import QDesktopServices, QIntValidator
from PyQt5.QtCore import QUrl
from preset_logic import apply_game_setup_preset, save_game_setup_preset_as, overwrite_game_setup_preset, reset_rules_shoe_to_defaults
import simulation_logic
# ---------- Reusable building blocks ----------

class SectionTitle(QtWidgets.QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("font-size:18px; font-weight:600; margin:2px 0 8px 0;")
        self.setWordWrap(True)


class SubTitle(QtWidgets.QLabel):
    def __init__(self, text):
        super().__init__()
        self.setText(text)
        self.setStyleSheet("font-size:14px; font-weight:600; margin:8px 0 4px 0;")


def make_line():
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line


def form_row(form: QtWidgets.QFormLayout, label: str, widget: QtWidgets.QWidget):
    lbl = QtWidgets.QLabel(label)
    lbl.setMinimumWidth(200)
    form.addRow(lbl, widget)


def pill(text: str):
    l = QtWidgets.QLabel(text)
    l.setStyleSheet("""
        QLabel {
            border: 1px solid #bdbdbd; border-radius: 10px; padding:2px 8px;
            background:#f5f5f5; color:#333;
        }
    """)
    return l


def button_row(*btns):
    row = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    lay.addStretch(1)
    for b in btns:
        lay.addWidget(b)
    return row

#----------- State Logic-----
class AppState(QtCore.QObject):
    config_changed = QtCore.pyqtSignal(object)  # emits Config

    def __init__(self):
        super().__init__()
        self._cfg = Config()

    def get_cfg(self) -> Config:
        return self._cfg

    def set_cfg(self, cfg: Config) -> None:
        self._cfg = cfg
        self.config_changed.emit(cfg)

# ---------- Pages ----------

class PageGameSetup(QtWidgets.QWidget):
    """
    Mirrors config_schema.Rules + Shoe (ALL fields).
    """
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.cfg = state.get_cfg()
        self.preset_path = "presets"
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        outer.addWidget(SectionTitle("Game Setup"))

        # ---- Presets ----
        preset_group = QtWidgets.QGroupBox("Presets")
        pg_lay = QtWidgets.QHBoxLayout(preset_group)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems([
            "Vegas Strip (S17, DAS, LS, 6 decks, 3:2)",
            "H17, No DAS, No Surrender, 8 decks, 3:2",
            "European No-Hole-Card (ENHC), 6 decks, 3:2",
            "Single Deck (S17), 3:2"
        ])
        apply_preset_btn = QtWidgets.QPushButton("Apply Preset")
        apply_preset_btn.clicked.connect(self.apply_preset)
        save_preset_btn = QtWidgets.QPushButton("Save as new Preset")
        save_preset_btn.clicked.connect(self.save_preset)
        overwrite_preset_btn = QtWidgets.QPushButton("Overwrite")
        overwrite_preset_btn.clicked.connect(self.overwrite_preset)
        default_preset_btn = QtWidgets.QPushButton("Apply default")
        default_preset_btn.clicked.connect(self.default_preset)
        pg_lay.addWidget(QtWidgets.QLabel("Preset:"))
        pg_lay.addWidget(self.preset_combo, 1)
        pg_lay.addStretch(1)
        pg_lay.addWidget(apply_preset_btn)
        pg_lay.addWidget(save_preset_btn)
        pg_lay.addWidget(overwrite_preset_btn)
        pg_lay.addWidget(default_preset_btn)
        outer.addWidget(preset_group)

        # ---- Rules (config_schema.Rules) ----
        rules = QtWidgets.QGroupBox("Rules")
        rf = QtWidgets.QFormLayout(rules)
        rf.setLabelAlignment(QtCore.Qt.AlignRight)

        self.blackjack_payout = QtWidgets.QComboBox()
        self.blackjack_payout.addItems(['3:2', '6:5'])
        self.blackjack_payout.setToolTip("rules.blackjack_payout")

        self.dealer_rule = QtWidgets.QComboBox()
        self.dealer_rule.addItems(['S17', 'H17'])
        self.dealer_rule.setToolTip("rules.dealer_rule")

        self.peek_rule = QtWidgets.QComboBox()
        self.peek_rule.addItems(['US', 'ENHC'])
        self.peek_rule.setToolTip("rules.peek_rule")

        self.push_22 = QtWidgets.QCheckBox("Push 22 (dealer 22 pushes)")
        self.push_22.setToolTip("rules.push_22")

        self.double_allowed = QtWidgets.QComboBox()
        self.double_allowed.addItems(['any_two', '10-11', '9-11'])
        self.double_allowed.setToolTip("rules.double_allowed")

        self.double_after_split = QtWidgets.QCheckBox("Allow Double After Split (DAS)")
        self.double_after_split.setChecked(True)
        self.double_after_split.setToolTip("rules.double_after_split")

        self.allow_splits = QtWidgets.QCheckBox("Allow Splits")
        self.allow_splits.setChecked(True)
        self.allow_splits.setToolTip("rules.allow_splits")

        self.max_splits = QtWidgets.QSpinBox()
        self.max_splits.setRange(0, 4)
        self.max_splits.setValue(3)
        self.max_splits.setToolTip("rules.max_splits")

        self.resplit_aces = QtWidgets.QCheckBox("Resplit Aces")
        self.resplit_aces.setToolTip("rules.resplit_aces")

        self.hit_split_aces = QtWidgets.QCheckBox("Hit Split Aces")
        self.hit_split_aces.setToolTip("rules.hit_split_aces")

        self.allow_surrender = QtWidgets.QComboBox()
        self.allow_surrender.addItems(['none', 'late', 'early'])
        self.allow_surrender.setToolTip("rules.allow_surrender")

        self.allow_insurance = QtWidgets.QCheckBox("Allow Insurance")
        self.allow_insurance.setToolTip("rules.allow_insurance")

        self.insurance_payout = QtWidgets.QComboBox()
        self.insurance_payout.addItems(['2:1'])
        self.insurance_payout.setToolTip("rules.insurance_payout")

        form_row(rf, "Blackjack Payout:", self.blackjack_payout)
        form_row(rf, "Dealer Rule:", self.dealer_rule)
        form_row(rf, "Peek Rule:", self.peek_rule)
        form_row(rf, "", self.push_22)
        form_row(rf, "Double Allowed:", self.double_allowed)
        form_row(rf, "", self.double_after_split)
        form_row(rf, "", self.allow_splits)
        form_row(rf, "Max Splits:", self.max_splits)
        form_row(rf, "", self.resplit_aces)
        form_row(rf, "", self.hit_split_aces)
        form_row(rf, "Surrender:", self.allow_surrender)
        form_row(rf, "", self.allow_insurance)
        form_row(rf, "Insurance Payout:", self.insurance_payout)

        # ---- Shoe (ALL fields from schema.Shoe) ----
        shoe = QtWidgets.QGroupBox("Shoe")
        sf = QtWidgets.QFormLayout(shoe)
        sf.setLabelAlignment(QtCore.Qt.AlignRight)

        self.n_decks = QtWidgets.QSpinBox()
        self.n_decks.setRange(1, 10)
        self.n_decks.setValue(6)
        self.n_decks.setToolTip("shoe.n_decks")

        pen_row = QtWidgets.QWidget()
        pen_lay = QtWidgets.QHBoxLayout(pen_row)
        pen_lay.setContentsMargins(0, 0, 0, 0)
        self.penetration_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.penetration_slider.setRange(50, 95)  # represent 0.50..0.95
        self.penetration_slider.setValue(75)      # 0.75 default
        self.penetration_slider.setToolTip("shoe.penetration (as %)")
        self.penetration_lbl = QtWidgets.QLabel("75%")
        self.penetration_slider.valueChanged.connect(lambda v: self.penetration_lbl.setText(f"{v}%"))
        pen_lay.addWidget(self.penetration_slider, 1)
        pen_lay.addWidget(self.penetration_lbl)

        self.burn_cards = QtWidgets.QSpinBox()
        self.burn_cards.setRange(0, 52)
        self.burn_cards.setValue(1)
        self.burn_cards.setToolTip("shoe.burn_cards")

        self.csm = QtWidgets.QCheckBox("Continuous Shuffling Machine (CSM)")
        self.csm.setToolTip("shoe.csm")

        self.shuffle_on_cutcard = QtWidgets.QCheckBox("Shuffle on cut-card")
        self.shuffle_on_cutcard.setChecked(True)
        self.shuffle_on_cutcard.setToolTip("shoe.shuffle_on_cutcard")

        self.penetration_variance = QtWidgets.QDoubleSpinBox()
        self.penetration_variance.setRange(0.0, 0.5)
        self.penetration_variance.setDecimals(3)
        self.penetration_variance.setSingleStep(0.005)
        self.penetration_variance.setValue(0.0)
        self.penetration_variance.setToolTip("shoe.penetration_variance")

        self.rng_seed = QtWidgets.QSpinBox()
        self.rng_seed.setRange(-1, 2**31 - 1)  # -1 means None
        self.rng_seed.setSpecialValueText("None")
        self.rng_seed.setValue(-1)
        self.rng_seed.setToolTip("shoe.rng_seed (optional)")

        form_row(sf, "Decks:", self.n_decks)
        form_row(sf, "Penetration:", pen_row)
        form_row(sf, "Burn cards:", self.burn_cards)
        form_row(sf, "", self.csm)
        form_row(sf, "", self.shuffle_on_cutcard)
        form_row(sf, "Penetration variance:", self.penetration_variance)
        form_row(sf, "RNG seed:", self.rng_seed)

        # Assemble
        outer.addWidget(rules)
        outer.addWidget(shoe)
        outer.addStretch(1)
        outer.addWidget(make_line())

        # Update the Widgets once
        self.refresh_preset_list()
        self.wire_game_setup_handlers()
        
    def refresh_preset_list(self, select_path: Path | None = None) -> None:
        """
        Scan self.preset_path and list preset files in the combo box.
        Shows names without the .yml/.yaml extension.
        """
        d = Path(self.preset_path)
        d.mkdir(parents=True, exist_ok=True)

        # collect presets; support both .yml and .yaml
        paths = sorted([*d.glob("*.yml"), *d.glob("*.yaml")], key=lambda p: p.name.lower())
        self._preset_paths = paths  # keep for Apply/Overwrite handlers

        names = [p.stem for p in paths]

        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(names)
        self.preset_combo.blockSignals(False)

        # optionally select a specific file
        if select_path:
            try:
                idx = paths.index(Path(select_path))
                self.preset_combo.setCurrentIndex(idx)
            except ValueError:
                pass 

    def apply_preset(self):
        preset_name = self.preset_combo.currentText()
        preset_path = os.path.join(self.preset_path, preset_name+".yaml")
        new_cfg = apply_game_setup_preset(self.cfg, preset_path)
        self.apply_config_to_widgets(new_cfg)

    def save_preset(self):
        """Prompt for a preset name and save current rules+shoe as a new YAML preset."""
        # get current config from widgets if available
        cfg = self.get_config() if hasattr(self, "get_config") else getattr(self, "cfg", None)
        if cfg is None:
            QtWidgets.QMessageBox.critical(self, "Save Preset", "No config loaded.")
            return

        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok:
            return  # user cancelled
        name = name.strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Save Preset", "Please enter a name.")
            return

        try:
            from preset_logic import save_game_setup_preset_as  # local import avoids circulars
            path = save_game_setup_preset_as(cfg, name, self.preset_path)  # self.preset_path = presets dir
        except FileExistsError:
            QtWidgets.QMessageBox.warning(
                self, "Save Preset",
                "A preset with that name already exists.\nUse the Overwrite button to replace it."
            )
            return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Preset", f"Failed to save preset:\n{e}")
            return

        # refresh preset list and select the new one, if helper exists
        if hasattr(self, "refresh_preset_list"):
            try:
                self.refresh_preset_list(select_path=path)
            except Exception:
                pass
        QtWidgets.QMessageBox.information(self, "Save Preset", f"Saved preset:\n{path}")

    def overwrite_preset(self):
        preset_name = self.preset_combo.currentText()
        preset_path = os.path.join(self.preset_path, preset_name+".yaml")
        overwrite_game_setup_preset(self.cfg, preset_path)
        QtWidgets.QMessageBox.information(self, "Overwrite Preset", f"{preset_name} successfully overwritten")

    def default_preset(self):
        new_cfg = reset_rules_shoe_to_defaults(self.cfg)
        self.apply_config_to_widgets(new_cfg)

    def _update_cfg(self, section: str, key: str, value):
        """Write-through update with validation."""
        base = self.cfg.model_dump(mode="python")
        base[section][key] = value
        self.cfg = Config(**base)
        self.state.set_cfg(self.cfg)

    def wire_game_setup_handlers(self):
        # --- Rules ---
        self.blackjack_payout.currentTextChanged.connect(
            lambda v: self._update_cfg("rules", "blackjack_payout", v)
        )
        self.dealer_rule.currentTextChanged.connect(
            lambda v: self._update_cfg("rules", "dealer_rule", v)
        )
        self.peek_rule.currentTextChanged.connect(
            lambda v: self._update_cfg("rules", "peek_rule", v)
        )
        self.push_22.toggled.connect(
            lambda b: self._update_cfg("rules", "push_22", bool(b))
        )
        self.double_allowed.currentTextChanged.connect(
            lambda v: self._update_cfg("rules", "double_allowed", v)
        )
        self.double_after_split.toggled.connect(
            lambda b: self._update_cfg("rules", "double_after_split", bool(b))
        )
        self.allow_splits.toggled.connect(
            lambda b: self._update_cfg("rules", "allow_splits", bool(b))
        )
        self.max_splits.valueChanged.connect(
            lambda v: self._update_cfg("rules", "max_splits", int(v))
        )
        self.resplit_aces.toggled.connect(
            lambda b: self._update_cfg("rules", "resplit_aces", bool(b))
        )
        self.hit_split_aces.toggled.connect(
            lambda b: self._update_cfg("rules", "hit_split_aces", bool(b))
        )
        self.allow_surrender.currentTextChanged.connect(
            lambda v: self._update_cfg("rules", "allow_surrender", v)
        )
        self.allow_insurance.toggled.connect(
            lambda b: self._update_cfg("rules", "allow_insurance", bool(b))
        )
        self.insurance_payout.currentTextChanged.connect(
            lambda v: self._update_cfg("rules", "insurance_payout", v)
        )

        # --- Shoe ---
        self.n_decks.valueChanged.connect(
            lambda v: self._update_cfg("shoe", "n_decks", int(v))
        )
        # slider is 50..95 representing 0.50..0.95
        self.penetration_slider.valueChanged.connect(
            lambda v: self._update_cfg("shoe", "penetration", float(v) / 100.0)
        )
        self.burn_cards.valueChanged.connect(
            lambda v: self._update_cfg("shoe", "burn_cards", int(v))
        )
        self.csm.toggled.connect(
            lambda b: self._update_cfg("shoe", "csm", bool(b))
        )
        self.shuffle_on_cutcard.toggled.connect(
            lambda b: self._update_cfg("shoe", "shuffle_on_cutcard", bool(b))
        )
        self.penetration_variance.valueChanged.connect(
            lambda v: self._update_cfg("shoe", "penetration_variance", float(v))
        )
        # -1 means None
        self.rng_seed.valueChanged.connect(
            lambda v: self._update_cfg("shoe", "rng_seed", None if int(v) == -1 else int(v))
        )

    def apply_config_to_widgets(self, cfg: Config) -> None:
        """Set all widgets from cfg.rules + cfg.shoe without emitting change signals."""
        self.cfg = cfg  # source of truth

        r = cfg.rules
        self._set_combo(self.blackjack_payout, r.blackjack_payout)
        self._set_combo(self.dealer_rule, r.dealer_rule)
        self._set_combo(self.peek_rule, r.peek_rule)
        self._set_check(self.push_22, r.push_22)
        self._set_combo(self.double_allowed, r.double_allowed)
        self._set_check(self.double_after_split, r.double_after_split)
        self._set_check(self.allow_splits, r.allow_splits)
        self._set_spin(self.max_splits, r.max_splits)
        self._set_check(self.resplit_aces, r.resplit_aces)
        self._set_check(self.hit_split_aces, r.hit_split_aces)
        self._set_combo(self.allow_surrender, r.allow_surrender)
        self._set_check(self.allow_insurance, r.allow_insurance)
        self._set_combo(self.insurance_payout, r.insurance_payout)

        s = cfg.shoe
        self._set_spin(self.n_decks, s.n_decks)
        self._set_slider_percent(self.penetration_slider, self.penetration_lbl, s.penetration)
        self._set_spin(self.burn_cards, s.burn_cards)
        self._set_check(self.csm, s.csm)
        self._set_check(self.shuffle_on_cutcard, s.shuffle_on_cutcard)
        self._set_doublespin(self.penetration_variance, float(s.penetration_variance))
        self._set_spin(self.rng_seed, -1 if s.rng_seed is None else int(s.rng_seed))

        self.state.set_cfg(self.cfg)

    def _set_combo(self, combo: QtWidgets.QComboBox, value: str) -> None:
        combo.blockSignals(True)
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _set_check(self, checkbox: QtWidgets.QCheckBox, checked: bool) -> None:
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(checked))
        checkbox.blockSignals(False)

    def _set_spin(self, spin: QtWidgets.QSpinBox, value: int) -> None:
        spin.blockSignals(True)
        spin.setValue(int(value))
        spin.blockSignals(False)

    def _set_doublespin(self, dspin: QtWidgets.QDoubleSpinBox, value: float) -> None:
        dspin.blockSignals(True)
        dspin.setValue(float(value))
        dspin.blockSignals(False)

    def _set_slider_percent(self, slider: QtWidgets.QSlider, label: QtWidgets.QLabel, frac: float) -> None:
        """frac in [0,1]; slider expects percent [50..95]."""
        val = int(round(frac * 100.0))
        slider.blockSignals(True)
        slider.setValue(val)
        slider.blockSignals(False)
        label.setText(f"{val}%")

class PageSimConsole(QtWidgets.QWidget):
    # default export folder: <this_file_dir>/sim_results
    DEFAULT_EXPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "sim_results"))
    
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.state.config_changed.connect(self._on_cfg)
        self._cfg = self.state.get_cfg()

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)
        outer.addWidget(SectionTitle("Simulation Console"))

        # --- Run parameters (TOP) ---
        params = QtWidgets.QGroupBox("Run parameters")
        pl = QtWidgets.QHBoxLayout(params)


        pl.addWidget(QtWidgets.QLabel("Episodes:"))
        self.episodes_spin = QtWidgets.QSpinBox()
        self.episodes_spin.setRange(1, 2_147_483_647)
        self.episodes_spin.setAccelerated(True)
        self.episodes_spin.setSingleStep(10_000)
        self.episodes_spin.setMaximumWidth(140)
        pl.addWidget(self.episodes_spin)

        pl.addSpacing(16)
        pl.addWidget(QtWidgets.QLabel("Random seed:"))
        self.seed_edit = QtWidgets.QLineEdit()
        self.seed_edit.setPlaceholderText("random")
        self.seed_edit.setValidator(QIntValidator())  # allow empty or int
        self.seed_edit.setMaximumWidth(180)
        pl.addWidget(self.seed_edit)
        pl.addStretch(1)
        outer.addWidget(params)

        # Prefill from cfg or defaults
        self.episodes_spin.setValue(int(self._cfg.simulation.episodes))
        self.seed_edit.setText("" if self._cfg.simulation.seed is None
                               else str(int(self._cfg.simulation.seed)))

        # --- Controls ---
        ctrl = QtWidgets.QGroupBox("Controls")
        cl = QtWidgets.QHBoxLayout(ctrl)
        start_btn = QtWidgets.QPushButton("Start")
        pause_btn = QtWidgets.QPushButton("Pause")
        stop_btn  = QtWidgets.QPushButton("Stop")
        cl.addWidget(start_btn); cl.addWidget(pause_btn); cl.addWidget(stop_btn)
        cl.addSpacing(12)
        cl.addWidget(QtWidgets.QLabel("Concurrent workers:"))
        workers = QtWidgets.QSpinBox(); workers.setRange(1, 64); workers.setValue(4)
        cl.addWidget(workers); cl.addStretch(1)
        outer.addWidget(ctrl)

        # --- Result export ---
        export = QtWidgets.QGroupBox("Result export")
        xl = QtWidgets.QHBoxLayout(export)
        xl.addWidget(QtWidgets.QLabel("Folder:"))
        self.export_edit = QtWidgets.QLineEdit(self.DEFAULT_EXPORT_DIR)
        self.export_edit.setMinimumWidth(360)
        browse = QtWidgets.QPushButton("Browse…")
        open_btn = QtWidgets.QPushButton("Open")
        xl.addWidget(self.export_edit, 1)
        xl.addWidget(browse)
        xl.addWidget(open_btn)
        outer.addWidget(export)

        browse.clicked.connect(self._choose_export_dir)
        open_btn.clicked.connect(self._open_export_dir)
        start_btn.clicked.connect(self.start_sim)

        # --- Progress ---
        prog = QtWidgets.QGroupBox("Progress")
        gl = QtWidgets.QGridLayout(prog)
        gl.addWidget(QtWidgets.QLabel("Overall:"), 0, 0)
        overall = QtWidgets.QProgressBar(); overall.setValue(0)
        gl.addWidget(overall, 0, 1)
        gl.addWidget(QtWidgets.QLabel("Hands/sec:"), 1, 0); gl.addWidget(pill("—"), 1, 1)
        gl.addWidget(QtWidgets.QLabel("EV/100 (running):"), 2, 0); gl.addWidget(pill("—"), 2, 1)
        gl.addWidget(QtWidgets.QLabel("95% CI width:"), 3, 0); gl.addWidget(pill("—"), 3, 1)
        outer.addWidget(prog)

        # --- Log ---
        log = QtWidgets.QGroupBox("Log")
        ll = QtWidgets.QVBoxLayout(log)
        t = QtWidgets.QPlainTextEdit(); t.setReadOnly(True)
        t.setPlaceholderText("[events will appear here]")
        ll.addWidget(t)
        outer.addWidget(log)
        outer.addStretch(1)

    def _on_cfg(self, cfg):
        self._cfg = cfg
        self.episodes_spin.setValue(int(cfg.simulation.episodes))
        self.seed_edit.setText("" if cfg.simulation.seed is None
                               else str(int(cfg.simulation.seed)))

    # --- export folder helpers ---
    def _effective_export_dir(self) -> str:
        txt = self.export_edit.text().strip()
        return os.path.abspath(os.path.expanduser(txt or self.DEFAULT_EXPORT_DIR))

    def _choose_export_dir(self):
        start_dir = self._effective_export_dir()
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select results folder", start_dir,
            options=QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        if d:
            self.export_edit.setText(d)

    def _open_export_dir(self):
        path = self._effective_export_dir()
        os.makedirs(path, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    # --- cfg IO for episodes/seed ---
    def _read_episodes_from_cfg(self, cfg) -> int:
        try:
            if hasattr(cfg, "simulation") and hasattr(cfg.simulation, "episodes"):
                return int(cfg.simulation.episodes)
        except Exception:
            pass
        return int(getattr(cfg, "episodes", 0) or 0)

    def _read_seed_from_cfg(self, cfg):
        try:
            if hasattr(cfg, "simulation") and hasattr(cfg.simulation, "seed"):
                return None if cfg.simulation.seed in (None, "") else int(cfg.simulation.seed)
        except Exception:
            pass
        val = getattr(cfg, "seed", None)
        return None if val in (None, "") else int(val)

    def _write_params_to_cfg(self, cfg, episodes: int, seed):
        wrote = False
        try:
            if hasattr(cfg, "simulation") and cfg.simulation is not None:
                if hasattr(cfg.simulation, "episodes"):
                    cfg.simulation.episodes = int(episodes); wrote = True
                if hasattr(cfg.simulation, "seed"):
                    cfg.simulation.seed = None if seed is None else int(seed); wrote = True
        except Exception:
            pass
        if not wrote:
            try:
                setattr(cfg, "episodes", int(episodes))
                setattr(cfg, "seed", None if seed is None else int(seed))
            except Exception:
                pass

    # --- simulation start ---
    def start_sim(self):
        cfg = self._cfg.model_copy(deep=True)

        # parameters -> cfg.simulation
        cfg.simulation.episodes = int(self.episodes_spin.value())
        seed_txt = self.seed_edit.text().strip()
        cfg.simulation.seed = None if seed_txt == "" else int(seed_txt)

        # export dir (unchanged)
        export_dir = self._effective_export_dir()
        os.makedirs(export_dir, exist_ok=True)
        if hasattr(cfg, "reporting") and hasattr(cfg.reporting, "output_dir"):
            cfg.reporting.output_dir = export_dir
        elif hasattr(cfg, "output_dir"):
            cfg.output_dir = export_dir

        simulation_logic.main(cfg, export_dir)


# ---------- Main Window ----------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blackjack Simulator")
        self.resize(1200, 800)
        self.state = AppState()
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        splitter = QtWidgets.QSplitter()
        splitter.setHandleWidth(6)
        splitter.setChildrenCollapsible(False)

        # Left navigation
        self.nav = QtWidgets.QListWidget()
        self.nav.setFixedWidth(260)
        self.nav.setSpacing(2)
        self.nav.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.nav.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.nav.addItems([
            "Game Setup",
            "Simulation Console"
        ])
        self.nav.setCurrentRow(0)

        # Right stacked pages
        self.stack = QtWidgets.QStackedWidget()
        self.pages = [
            PageGameSetup(self.state),
            PageSimConsole(self.state)
        ]
        for p in self.pages:
            self.stack.addWidget(p)

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)

        splitter.addWidget(self.nav)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # Menus & toolbar (visual only)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("New")
        file_menu.addAction("Open…")
        file_menu.addAction("Save")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction("Preferences…")

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("About")

        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.addAction("New")
        tb.addAction("Open")
        tb.addAction("Save")

        # Status bar
        sb = QtWidgets.QStatusBar()
        sb.showMessage("Ready")
        self.setStatusBar(sb)
        self.state.config_changed.connect(lambda cfg: self.statusBar().showMessage("Config updated"))   #Show config updates

    def _apply_style(self):
        self.setStyleSheet("""
        QMainWindow { background: #ffffff; }
        QListWidget {
            border: none; background: #f7f7f9; padding: 8px 6px;
            font-size: 14px;
        }
        QListWidget::item {
            padding: 10px 8px; border-radius: 8px; margin: 2px 0;
        }
        QListWidget::item:selected {
            background: #e3f2fd; color: #0d47a1;
        }
        QGroupBox {
            border: 1px solid #e0e0e0; border-radius: 8px; margin-top: 12px; padding: 8px 12px 12px 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #424242;
            background: transparent;
        }
        QTableWidget {
            gridline-color: #e0e0e0;
            selection-background-color: #bbdefb;
        }
        QProgressBar {
            border: 1px solid #d0d0d0; border-radius: 6px; text-align: center; height: 16px;
        }
        QProgressBar::chunk { background-color: #64b5f6; }
        QPushButton {
            padding: 6px 12px; border: 1px solid #cfd8dc; border-radius: 8px; background: #fafafa;
        }
        QPushButton:hover { background: #f0f0f0; }
        QLabel { color: #333; }
        QStatusBar { background: #fafafa; }
        """)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Blackjack Simulator — Prototype")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
