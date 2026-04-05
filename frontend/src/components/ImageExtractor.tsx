import { useRef, useState, useEffect, type MouseEvent, type TouchEvent } from "react";

interface Rect { x1: number; y1: number; x2: number; y2: number; }
interface Props { onExtract: (question: string, options: string[]) => void; }

function canvasCoords(
  clientX: number, clientY: number,
  canvas: HTMLCanvasElement
): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.round((clientX - rect.left) * (canvas.width / rect.width)),
    y: Math.round((clientY - rect.top) * (canvas.height / rect.height)),
  };
}

export default function ImageExtractor({ onExtract }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imageData, setImageData] = useState<string | null>(null);
  const [selection, setSelection] = useState<Rect | null>(null);
  const [drawing, setDrawing] = useState(false);
  const pending = useRef<Rect>({ x1: 0, y1: 0, x2: 0, y2: 0 });
  const [numOptions, setNumOptions] = useState(4);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadImage = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const src = e.target!.result as string;
      const img = new Image();
      img.onload = () => {
        imgRef.current = img;
        setImageData(src);   // triggers re-render → canvas mounts → useEffect draws
        setSelection(null);
      };
      img.src = src;
    };
    reader.readAsDataURL(file);
  };

  // Draw image after canvas mounts (canvas only exists once imageData is set)
  useEffect(() => {
    if (!imageData || !canvasRef.current || !imgRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = imgRef.current.width;
    canvas.height = imgRef.current.height;
    canvas.getContext("2d")!.drawImage(imgRef.current, 0, 0);
  }, [imageData]);

  const redraw = (rect: Rect) => {
    if (!imgRef.current || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d")!;
    ctx.drawImage(imgRef.current, 0, 0);
    ctx.strokeStyle = "#0ea5e9"; ctx.lineWidth = 2;
    ctx.strokeRect(rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1);
    ctx.fillStyle = "rgba(14,165,233,0.1)";
    ctx.fillRect(rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1);
  };

  const startDraw = (x: number, y: number) => {
    pending.current = { x1: x, y1: y, x2: x, y2: y };
    setDrawing(true); setSelection(null);
  };
  const moveDraw = (x: number, y: number) => {
    if (!drawing) return;
    pending.current.x2 = x; pending.current.y2 = y;
    redraw(pending.current);
  };
  const endDraw = () => {
    if (!drawing) return;
    setDrawing(false); setSelection({ ...pending.current });
  };

  const onMouseDown = (e: MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = canvasCoords(e.clientX, e.clientY, canvasRef.current!);
    startDraw(x, y);
  };
  const onMouseMove = (e: MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = canvasCoords(e.clientX, e.clientY, canvasRef.current!);
    moveDraw(x, y);
  };
  const onTouchStart = (e: TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const t = e.touches[0];
    const { x, y } = canvasCoords(t.clientX, t.clientY, canvasRef.current!);
    startDraw(x, y);
  };
  const onTouchMove = (e: TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const t = e.touches[0];
    const { x, y } = canvasCoords(t.clientX, t.clientY, canvasRef.current!);
    moveDraw(x, y);
  };

  const extract = async () => {
    if (!imageData || !selection) return;
    setLoading(true); setError("");
    try {
      const form = new FormData();
      form.append("image_data", imageData);
      form.append("x1", String(Math.min(selection.x1, selection.x2)));
      form.append("y1", String(Math.min(selection.y1, selection.y2)));
      form.append("x2", String(Math.max(selection.x1, selection.x2)));
      form.append("y2", String(Math.max(selection.y1, selection.y2)));
      form.append("num_options", String(numOptions));

      const res = await fetch("/api/v1/ocr/extract", { method: "POST", body: form });
      if (!res.ok) throw new Error("OCR failed");
      const data = await res.json();
      onExtract(data.question, data.options);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Extraction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-6 mt-4">
      <p className="text-sm font-medium text-slate-700 mb-3">Extract from image</p>

      <label className="block cursor-pointer">
        <input
          type="file"
          accept="image/*"
          capture="environment"
          className="hidden"
          onChange={e => e.target.files?.[0] && loadImage(e.target.files[0])}
        />
        <div className="border-2 border-dashed border-slate-200 rounded-xl p-4 text-center text-sm text-slate-400 hover:border-brand-500 transition-colors">
          {imageData ? "Tap to change image" : "Tap to upload or take photo"}
        </div>
      </label>

      {imageData && (
        <>
          <canvas
            ref={canvasRef}
            className="w-full mt-3 rounded-lg border border-slate-200 cursor-crosshair touch-none"
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={endDraw}
            onTouchStart={onTouchStart}
            onTouchMove={onTouchMove}
            onTouchEnd={endDraw}
          />
          <div className="flex items-center gap-3 mt-3">
            <label className="text-sm text-slate-600">Options:</label>
            <select value={numOptions} onChange={e => setNumOptions(Number(e.target.value))}
              className="border border-slate-200 rounded-lg px-2 py-1 text-sm">
              {[2, 3, 4].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
            <button onClick={extract} disabled={!selection || loading}
              className="flex-1 bg-brand-500 text-white text-sm py-2 rounded-lg disabled:opacity-40 transition-colors">
              {loading ? "Extracting…" : "Extract Text"}
            </button>
          </div>
          {error && <p className="text-red-600 text-sm mt-2">{error}</p>}
        </>
      )}
    </div>
  );
}
