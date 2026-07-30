import sys
from pathlib import Path
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rag_core
query = "Que correlativas tiene Auditoria y Peritaje de Sistemas Informaticos"
res = rag_core.detect_asignaturas_query(query)
print("Detected:", res)
