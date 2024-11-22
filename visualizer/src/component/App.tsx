import derivedComparator from "@xtjs/lib/derivedComparator";
import distinctFilter from "@xtjs/lib/distinctFilter";
import exists from "@xtjs/lib/exists";
import map from "@xtjs/lib/map";
import mapExists from "@xtjs/lib/mapExists";
import min from "@xtjs/lib/min";
import propertyComparator from "@xtjs/lib/propertyComparator";
import shuffleArray from "@xtjs/lib/shuffleArray";
import sum from "@xtjs/lib/sum";
import { useEffect, useMemo, useRef, useState } from "react";
import seedrandom from "seedrandom";
import { dist, Graph, GraphNode, Point } from "../common";
import "./App.css";
import { GraphSvg } from "./Graph";

// This implementation is intentionally slow and verbose to keep as close to the reference paper description as possible. (We don't need fast performance for this project anyway.)
const greedySearch = (g: Graph, q: Point, k: number, searchListCap: number) => {
  const searchList = Array<{ id: number; dist: number; from: number }>();
  searchList.push({
    id: g.medoid,
    dist: dist(g.getPoint(g.medoid), q),
    from: -1,
  });
  const visited = new Set<number>();
  const edges = Array<{ source: number; target: number }>();
  while (true) {
    const pStar = searchList.find((p) => !visited.has(p.id));
    if (!pStar) {
      break;
    }
    for (const n of g.getNeighbors(pStar.id)) {
      if (!searchList.find((p) => p.id === n)) {
        searchList.push({
          id: n,
          dist: dist(g.getPoint(n), q),
          from: pStar.id,
        });
      }
    }
    visited.add(pStar.id);
    if (pStar.from !== -1) {
      edges.push({ source: pStar.from, target: pStar.id });
    }
    searchList.sort(propertyComparator("dist")).splice(searchListCap);
  }
  return [searchList.slice(0, k), visited, edges] as const;
};

// This implementation is intentionally slow and verbose to keep as close to the reference paper description as possible. (We don't need fast performance for this project anyway.)
const computeRobustPruned = (
  g: Graph,
  p: number,
  candidates: Array<number>,
  distanceThreshold: number,
  degreeBound: number,
) => {
  candidates = [...candidates, ...g.getNeighbors(p)]
    .filter(distinctFilter())
    .filter((c) => c !== p)
    .sort(derivedComparator((c) => g.dist(c, p)));
  const newNeighbors = Array<number>();
  while (candidates.length) {
    const pStar = candidates.shift()!;
    newNeighbors.push(pStar);
    if (newNeighbors.length === degreeBound) {
      break;
    }
    candidates = candidates.filter(
      (pPrime) =>
        !(distanceThreshold * g.dist(pStar, pPrime) <= g.dist(p, pPrime)),
    );
  }
  return newNeighbors;
};

const optimizedGraph = (
  og: Graph,
  searchListCap: number,
  distanceThreshold: number,
  degreeBound: number,
) => {
  const ng = og.clone();
  for (const id of shuffleArray([...ng.nodes.keys()])) {
    const [_, visited] = greedySearch(ng, ng.getPoint(id), 1, searchListCap);
    const newNeighbors = computeRobustPruned(
      ng,
      id,
      [...visited],
      distanceThreshold,
      degreeBound,
    );
    ng.nodes.get(id)!.neighbors = newNeighbors;
    for (const j of newNeighbors) {
      const jNeighbors = ng.getNeighbors(j);
      if (jNeighbors.includes(id)) {
        continue;
      }
      jNeighbors.push(id);
      if (jNeighbors.length > degreeBound) {
        const newJNeighbors = computeRobustPruned(
          ng,
          j,
          jNeighbors,
          distanceThreshold,
          degreeBound,
        );
        ng.nodes.get(j)!.neighbors = newJNeighbors;
      }
    }
  }
  return ng;
};

const initRandomGraph = ({
  seed,
  maxX,
  maxY,
  neighborCount,
  nodeCount,
}: {
  seed: string;
  maxX: number;
  maxY: number;
  neighborCount: number;
  nodeCount: number;
}) => {
  const rng = seedrandom(seed);
  const nodes = new Map<number, GraphNode>();
  for (let i = 0; i < nodeCount; i++) {
    const point = { x: rng() * maxX, y: rng() * maxY };
    const neighbors = new Set<number>();
    while (neighbors.size < neighborCount) {
      neighbors.add(Math.floor(rng() * nodeCount));
    }
    nodes.set(i, { point, neighbors: Array.from(neighbors) });
  }
  const medoid = min(
    nodes,
    derivedComparator(([_, n]) =>
      sum(map(nodes.values(), (other) => dist(n.point, other.point))),
    ),
  )[0];
  return new Graph(nodes, medoid);
};

const Slider = ({
  value,
  onChange,
  min,
  max,
  step,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) => (
  <input
    type="range"
    value={value}
    onChange={(e) => onChange(e.target.valueAsNumber)}
    min={min}
    max={max}
    step={step}
  />
);

export const App = () => {
  const $graphContainer = useRef<HTMLDivElement>(null);
  const [seed, setSeed] = useState("random seed");
  const [nodeCount, setNodeCount] = useState(200);
  const [height, setHeight] = useState(900);
  const [width, setWidth] = useState(1600);
  const [querySearchListCap, setQuerySearchListCap] = useState(1);
  const [updateSearchListCap, setUpdateSearchListCap] = useState(10);
  const [distanceThreshold, setDistanceThreshold] = useState(1.1);
  const [degreeBound, setDegreeBound] = useState(10);
  const [graph, setGraph] = useState(() =>
    initRandomGraph({
      seed,
      maxX: width,
      maxY: height,
      neighborCount: degreeBound,
      nodeCount,
    }),
  );
  const [query, setQuery] = useState<Point>();
  const result = useMemo(
    () => query && greedySearch(graph, query, 1, querySearchListCap),
    [graph, query, querySearchListCap],
  );
  const actualAnswer = useMemo(
    () =>
      query &&
      min(
        graph.nodes,
        derivedComparator(([_, n]) => dist(n.point, query)),
      )[0],
    [graph, query],
  );
  const correct = actualAnswer === result?.[0][0].id;

  useEffect(() => {
    if (!$graphContainer.current) {
      return;
    }
    const resize = () => {
      if ($graphContainer.current) {
        setHeight($graphContainer.current.clientHeight);
        setWidth($graphContainer.current.clientWidth);
      }
    };
    const observer = new ResizeObserver(resize);
    observer.observe($graphContainer.current!);
    resize();
    return () => observer.disconnect();
  }, []);

  return (
    <div className="App">
      <div ref={$graphContainer} className="graph-container">
        <GraphSvg
          edges={graph.edges()}
          nodes={[
            ...map(graph.nodes, ([id, node]) => ({ id, point: node.point })),
            query && { id: -1, point: query },
          ].filter(exists)}
          height={height}
          width={width}
          onClick={setQuery}
          nodeStyles={[
            { id: -1, className: "node-query", r: 4 },
            { id: graph.medoid, className: "node-medoid", r: 4 },
            result && {
              id: result?.[0][0].id,
              className: correct
                ? "node-result-correct"
                : "node-result-incorrect",
              r: 4,
            },
            ...(result?.[2].map((r, i) => ({
              id: r.target,
              className: "node-query-visited",
            })) ?? []),
          ].filter(exists)}
          edgeStyles={[
            ...(result?.[2].map(({ source, target }, i) => ({
              source,
              target,
              className: "edge-query",
              animationDelay: `${i * 60}ms`,
            })) ?? []),
          ]}
        />
      </div>
      <div className="panel">
        <div className="buttons">
          <button
            onClick={() => {
              setGraph(
                initRandomGraph({
                  seed,
                  nodeCount,
                  neighborCount: degreeBound,
                  maxX: width,
                  maxY: height,
                }),
              );
            }}
          >
            Random
          </button>
          <button
            onClick={() => {
              setGraph(
                optimizedGraph(
                  graph,
                  updateSearchListCap,
                  distanceThreshold,
                  degreeBound,
                ),
              );
            }}
          >
            Optimize
          </button>
        </div>

        <label>
          <span>Random seed</span>
          <div>
            <input value={seed} onChange={(e) => setSeed(e.target.value)} />
          </div>
        </label>

        <label>
          <span>Node count</span>
          <div>
            <Slider
              value={nodeCount}
              onChange={setNodeCount}
              min={1}
              max={1000}
            />
            <span>{nodeCount}</span>
          </div>
        </label>

        <label>
          <span>Degree bound</span>
          <div>
            <Slider
              value={degreeBound}
              onChange={setDegreeBound}
              min={1}
              max={20}
            />
            <span>{degreeBound}</span>
          </div>
        </label>

        <label>
          <span>Query search list cap</span>
          <div>
            <Slider
              value={querySearchListCap}
              onChange={setQuerySearchListCap}
              min={1}
              max={20}
            />
            <span>{querySearchListCap}</span>
          </div>
        </label>

        <label>
          <span>Update search list cap</span>
          <div>
            <Slider
              value={updateSearchListCap}
              onChange={setUpdateSearchListCap}
              min={1}
              max={20}
            />
            <span>{updateSearchListCap}</span>
          </div>
        </label>

        <label>
          <span>Distance threshold</span>
          <div>
            <Slider
              value={distanceThreshold}
              onChange={setDistanceThreshold}
              min={1}
              max={4}
              step={0.01}
            />
            <span>{distanceThreshold}</span>
          </div>
        </label>

        {mapExists(result, (r) => (
          <div className="result">
            <h3>
              Result is{" "}
              <span className={correct ? "correct" : "incorrect"}></span>
            </h3>
            <div>Visited {r[1].size} nodes</div>
          </div>
        ))}
      </div>
    </div>
  );
};
