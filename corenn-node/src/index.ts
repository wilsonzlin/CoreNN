let internal: any;
try {
  // Prioritise any local built binary.
  internal = require("./index.node");
} catch {
  internal = require(`@corenn/node-${process.platform}-${process.arch}`);
}

export type Cfg = {
  dim: number;
} & Partial<{
  beamWidth: number;
  compactionThresholdDeletes: number;
  compressionMode: "pq" | "trunc";
  compressionThreshold: number;
  distanceThreshold: number;
  maxAddEdges: number;
  maxEdges: number;
  metric: "l2" | "cosine";
  pqSampleSize: number;
  pqSubspaces: number;
  querySearchListCap: number;
  truncDims: number;
  updateSearchListCap: number;
}>;

export class CoreNN {
  private constructor(private readonly db: any) {}

  static create(path: string, cfg: Cfg) {
    const db = internal.create_db(path, cfg);
    return new CoreNN(db);
  }

  static open(path: string) {
    const db = internal.open_db(path);
    return new CoreNN(db);
  }

  static newInMemory(cfg: Cfg) {
    const db = internal.new_in_memory(cfg);
    return new CoreNN(db);
  }

  insert(key: string, vector: Float32Array | Float64Array) {
    internal.insert(this.db, key, vector);
  }

  query(query: Float32Array | Float64Array, k: number): Array<{ key: string; distance: number }> {
    return internal.query(this.db, query, k);
  }
}
