const fs = require('fs');
const path = require('path');

const notebookPath = path.join(__dirname, 'TickNets_CBAM_Classification.ipynb');
const notebook = JSON.parse(fs.readFileSync(notebookPath, 'utf8'));

// Fix print statements in all cells
notebook.cells.forEach((cell, idx) => {
  if (cell.cell_type === 'code' && cell.source) {
    const newSource = [];
    for (let i = 0; i < cell.source.length; i++) {
      let line = cell.source[i];
      
      // 1. Handle print("\n" + "="*60)
      if (line.includes('print("\\n') && line.includes('="*60')) {
        newSource.push('print()\n');
        newSource.push('print("="*60)\n');
        continue;
      }
      
      // 2. Handle general print("\n...") or print(f"\n...")
      if (line.includes('print("') && line.includes('\\n') && !line.includes('print("\\n"')) {
        // If it starts with print("\n or print(f"\n
        if (line.includes('print("\\n') || line.includes('print(f"\\n')) {
          newSource.push('print()\n');
          line = line.replace('print("\\n', 'print("').replace('print(f"\\n', 'print(f"');
          newSource.push(line);
          continue;
        }
      }
      
      newSource.push(line);
    }
    
    // Also clean up any split lines if they occurred in previous runs
    const finalSource = [];
    for (let i = 0; i < newSource.length; i++) {
      const line = newSource[i];
      // If we see a line like print("\n and next line is " + "="*60)
      if (line === 'print("\n' && i + 1 < newSource.length && newSource[i+1].includes('" + "="*60)')) {
        finalSource.push('print()\n');
        finalSource.push('print("="*60)\n');
        i++; // skip next line
        continue;
      }
      finalSource.push(line);
    }
    
    cell.source = finalSource;
  }
});

fs.writeFileSync(notebookPath, JSON.stringify(notebook, null, 2), 'utf8');
console.log("Notebook prints patched successfully!");
