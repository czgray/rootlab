# Copyright 2025 Christian Gray
#
# File: io.py
#
# Author: Christian Gray (czgray@princeton.edu)
# 
# RootLab is a free, open-source package developed by Christian Gray
# as part of his Ph.D. research at Princeton University.
# It is designed to aid in the analysis of plant root architecture.
# If you use this package in your work, please cite it.
# See the "NOTICE" file for citation details.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""I/O helpers for JSON and related artifacts."""
from pathlib import Path
import json
from typing import Any

def load_json(path: str | Path) -> Any:
    """Load a JSON file into a Python object."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Save a Python object to JSON."""
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
