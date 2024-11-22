import { Edge, Point } from "../common";
import "./Graph.css";

export const GraphSvg = ({
  defaultEdgeClass = "default-edge",
  defaultNodeClass = "default-node",
  edgeStyles,
  edges,
  height,
  nodeStyles,
  nodes,
  onClick,
  width,
}: {
  defaultEdgeClass?: string;
  defaultNodeClass?: string;
  edgeStyles?: Array<{
    source: number;
    target: number;
    className: string;
    animationDelay?: string;
  }>;
  edges: Array<Edge>;
  height: number;
  nodeStyles?: Array<{
    id: number;
    className: string;
    r?: number;
    animationDelay?: string;
  }>;
  nodes: Array<{ id: number; point: Point }>;
  onClick?: (pt: Point) => void;
  width: number;
}) => {
  return (
    <svg
      className="Graph"
      height={height}
      width={width}
      onClick={(e) => {
        const svg = e.currentTarget;
        const rect = svg.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        onClick?.({ x, y });
      }}
    >
      {/* Draw edges. */}
      {edges.map(({ source: sid, target: tid }) => {
        const sourcePt = nodes.find(({ id }) => id === sid)!.point;
        const targetPt = nodes.find(({ id }) => id === tid)!.point;
        const sty = edgeStyles?.find(
          ({ source, target }) => source === sid && target === tid,
        );
        return (
          <line
            key={`edge-${sid}-${tid}`}
            className={[sty?.className, defaultEdgeClass].join(" ")}
            x1={sourcePt.x}
            y1={sourcePt.y}
            x2={targetPt.x}
            y2={targetPt.y}
            style={{ animationDelay: sty?.animationDelay }}
          />
        );
      })}

      {/* Draw nodes. */}
      {nodes.map(({ id, point }) => {
        const sty = nodeStyles?.find((n) => n.id === id);
        return (
          <circle
            key={id}
            className={[sty?.className, defaultNodeClass].join(" ")}
            cx={point.x}
            cy={point.y}
            r={sty?.r ?? 3}
            style={{ animationDelay: sty?.animationDelay }}
          />
        );
      })}
    </svg>
  );
};
