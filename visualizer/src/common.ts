import map from "@xtjs/lib/map";

export const dist = (a: Point, b: Point) => {
  return Math.hypot(a.x - b.x, a.y - b.y);
};

export type Edge = {
  source: number;
  target: number;
};

export type Point = {
  x: number;
  y: number;
};

export type GraphNode = {
  point: Point;
  neighbors: number[];
};

export class Graph {
  constructor(
    readonly nodes: Map<number, GraphNode>,
    readonly medoid: number,
  ) {}

  clone() {
    return new Graph(
      new Map(
        map(
          this.nodes,
          ([id, node]) =>
            [
              id,
              {
                point: node.point,
                neighbors: [...node.neighbors],
              },
            ] as const,
        ),
      ),
      this.medoid,
    );
  }

  dist(a: number, b: number) {
    return dist(this.getPoint(a), this.getPoint(b));
  }

  edges() {
    return [...this.nodes].flatMap(([sid, source]) =>
      source.neighbors.map((tid) => ({ source: sid, target: tid })),
    );
  }

  getPoint(id: number): Point {
    return this.nodes.get(id)!.point;
  }

  getNeighbors(id: number): number[] {
    return this.nodes.get(id)!.neighbors;
  }
}
