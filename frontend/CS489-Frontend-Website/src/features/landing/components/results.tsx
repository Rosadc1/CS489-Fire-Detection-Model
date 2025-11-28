import { useState, useRef, useEffect } from "react";
import { AlertTriangle, Flame, Cloud, Loader2, CheckCircle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type DetectionResult = {
  classification: "fire" | "no_fire";
  confidence: number;
  localization?: {
    image: string;
    detectedObjects: Array<{
      name: string;
      confidence: number;
      box: {
        x1: number;
        y1: number;
        x2: number;
        y2: number;
      };
    }>;
  };
};

type ProcessingStage = "classifying" | "localizing" | "complete" | null;

type ResultsContainerProps = {
  selectedImage: string;
  result: DetectionResult | null;
  processingStage: ProcessingStage;
  error: string;
};

export function ResultsContainer({
  selectedImage,
  result,
  processingStage,
  error,
}: ResultsContainerProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const img = imageRef.current;
    if (img && img.complete && img.naturalWidth > 0) {
      setImageLoaded(true);
      setImageDimensions({ width: img.width, height: img.height });
    }
  }, [selectedImage]);

  const handleImageLoad = () => {
    const img = imageRef.current;
    if (img) {
      setImageLoaded(true);
      setImageDimensions({ width: img.width, height: img.height });
    }
  };

  const getClassificationIcon = (classification: string) => {
    switch (classification) {
      case "fire":
        return <Flame className="w-4 h-4" />;
      case "smoke":
        return <Cloud className="w-4 h-4" />;
      default:
        return <AlertTriangle className="w-4 h-4" />;
    }
  };

  return (
    <>
      {error && (
        <Card className="p-6 shadow-lg border-2 border-red-300 bg-red-50">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-6 h-6 text-red-600" />
            <div>
              <p className="font-semibold text-red-900">Analysis Error</p>
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </Card>
      )}

      {selectedImage && (
        <Card className="p-8 shadow-2xl border-0 bg-white/80 backdrop-blur-sm overflow-hidden">
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-center bg-linear-to-r from-orange-600 to-red-600 bg-clip-text text-transparent">
              Analysis Results
            </h2>

            {(processingStage === "classifying" || processingStage === "localizing") && (
              <div className="flex flex-col items-center gap-4 py-8">
                <Loader2 className={`w-12 h-12 animate-spin ${
                  processingStage === "classifying" ? "text-orange-500" : "text-red-500"
                }`} />
                <p className="text-lg font-medium text-gray-700">
                  {processingStage === "classifying" ? "Classifying image..." : "Localizing fire origin..."}
                </p>
            
              </div>
            )}

            {processingStage === "complete" && result && result.classification === "no_fire" && (
              <div className="space-y-6 animate-fade-in">
                <div className="relative w-full rounded-xl overflow-hidden shadow-2xl">
                  <img
                    src={selectedImage}
                    alt="Analyzed image"
                    className="w-full h-auto"
                  />
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between p-5 bg-linear-to-br from-green-50 to-emerald-50 rounded-xl shadow-md border-2 border-green-200">
                    <div className="flex items-center gap-3">
                      <div className="bg-green-100 p-3 rounded-lg">
                        <CheckCircle className="w-4 h-4 text-green-600" />
                      </div>
                      <span className="font-semibold text-gray-800">Status</span>
                    </div>
                    <Badge className="bg-linear-to-r from-green-500 to-emerald-500 text-white capitalize shadow-lg px-4 py-2">
                      Safe
                    </Badge>
                  </div>

                  <div className="flex items-center justify-between p-5 bg-linear-to-br from-gray-50 to-gray-100 rounded-xl shadow-md border border-gray-200">
                    <div className="flex items-center gap-3">
                      <div className="bg-blue-100 p-3 rounded-lg">
                        <AlertTriangle className="w-4 h-4 text-blue-600" />
                      </div>
                      <span className="font-semibold text-gray-800">Confidence Score</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-linear-to-r from-green-500 to-emerald-500 rounded-full transition-all duration-1000"
                          style={{ width: `${result.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="font-bold text-gray-900 px-3 py-1 bg-white rounded-lg shadow-sm min-w-16 text-center">
                        {(result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="p-6 bg-linear-to-br from-green-50 to-emerald-50 border-2 border-green-300 rounded-xl shadow-lg">
                    <div className="flex items-center gap-3">
                      <CheckCircle className="w-6 h-6 text-green-600" />
                      <div>
                        <p className="font-bold text-green-900">No Threat Detected</p>
                        <p className="text-sm text-green-800 mt-1">
                          The model found no signs of fire or smoke in this image.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {processingStage === "complete" && result && result.classification === "fire" && result.localization && (
              <div className="space-y-6 animate-fade-in">
                <div className="relative w-full rounded-xl overflow-visible shadow-2xl bg-gray-100 ring-4 ring-orange-100">
                  <img
                    ref={imageRef}
                    src={selectedImage}
                    alt="Original uploaded image"
                    className="w-full h-auto"
                    onLoad={handleImageLoad}
                  />
                  
                  {imageLoaded && imageDimensions.width > 0 && imageRef.current && result.localization.detectedObjects.map((obj, idx) => {
                    const img = imageRef.current!;
                
                    // Calculate dimenions
                    const naturalWidth = img.naturalWidth;
                    const naturalHeight = img.naturalHeight;
                    const displayedWidth = img.width;
                    const displayedHeight = img.height;
                    
                    // Scale dimensions
                    const scaleX = displayedWidth / naturalWidth;
                    const scaleY = displayedHeight / naturalHeight;
                    
                    const scaledLeft = obj.box.x1 * scaleX;
                    const scaledTop = obj.box.y1 * scaleY;
                    const scaledWidth = (obj.box.x2 - obj.box.x1) * scaleX;
                    const scaledHeight = (obj.box.y2 - obj.box.y1) * scaleY;
     
                    return (
                      <div
                        key={idx}
                        className="absolute border-4 rounded-lg"
                        style={{
                          left: `${scaledLeft}px`,
                          top: `${scaledTop}px`,
                          width: `${scaledWidth}px`,
                          height: `${scaledHeight}px`,
                          borderColor: obj.name === 'fire' ? '#FFD700' : '#6b7280',
                          boxShadow: obj.name === 'fire' 
                            ? '0 0 20px rgba(255, 215, 0, 0.8)' 
                            : '0 0 20px rgba(107, 114, 128, 0.8)',
                          zIndex: 999,
                          pointerEvents: 'none',
                        }}
                      >
                         <div 
                          className="absolute -top-9 left-0 px-3 py-1.5 rounded-lg shadow-lg text-sm font-semibold text-white whitespace-nowrap"
                          style={{
                            background: obj.name === 'fire' 
                              ? 'linear-gradient(to right, #ef4444, #f97316)' 
                              : 'linear-gradient(to right, #6b7280, #4b5563)'
                          }}
                        >
                          {obj.name === 'fire' ? (
                            <><Flame className="w-3 h-3 inline mr-1" /> Fire {(obj.confidence * 100).toFixed(0)}%</>
                          ) : (
                            <><Cloud className="w-3 h-3 inline mr-1" /> Smoke {(obj.confidence * 100).toFixed(0)}%</>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between p-5 bg-linear-to-br from-red-50 to-orange-50 rounded-xl shadow-md border-2 border-red-200">
                    <div className="flex items-center gap-3">
                      <div className="bg-red-100 p-3 rounded-lg">
                        {getClassificationIcon(result.classification)}
                      </div>
                      <span className="font-semibold text-gray-800">Threat Detected</span>
                    </div>
                    <Badge className="bg-linear-to-r from-red-500 to-orange-500 text-white capitalize shadow-lg px-4 py-2">
                      Fire
                    </Badge>
                  </div>

                  <div className="flex items-center justify-between p-5 bg-linear-to-br from-gray-50 to-gray-100 rounded-xl shadow-md border border-gray-200">
                    <div className="flex items-center gap-3">
                      <div className="bg-blue-100 p-3 rounded-lg">
                        <AlertTriangle className="w-4 h-4 text-blue-600" />
                      </div>
                      <span className="font-semibold text-gray-800">Confidence Score</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-linear-to-r from-orange-500 to-red-500 rounded-full transition-all duration-1000"
                          style={{ width: `${result.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="font-bold text-gray-900 px-3 py-1 bg-white rounded-lg shadow-sm min-w-16 text-center">
                        {(result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {result.localization.detectedObjects.length > 0 && (
                    <div className="p-6 bg-linear-to-br from-orange-50 to-red-50 border-2 border-orange-300 rounded-xl shadow-lg">
                      <div className="flex items-start gap-3 mb-4">
                        <div className="bg-orange-100 p-2 rounded-lg mt-0.5">
                          <AlertTriangle className="w-6 h-6 text-orange-600" />
                        </div>
                        <div className="flex-1">
                          <p className="font-bold text-orange-900 mb-1">
                            Warning: {result.localization.detectedObjects.length} Object(s) Detected
                          </p>
                          <p className="text-sm text-orange-800">
                            Detected fire areas are highlighted by the boxes in the image.
                          </p>
                        </div>
                      </div>
                      
                      <div className="space-y-2 mt-4">
                        {result.localization.detectedObjects.map((obj, idx) => (
                          <div key={idx} className="bg-white/60 p-3 rounded-lg flex justify-between items-center">
                            <div className="flex items-center gap-2">
                              <Badge className={obj.name === 'fire' ? 'bg-red-500' : 'bg-gray-500'}>
                                {obj.name.toUpperCase()}
                              </Badge>
                            </div>
                            <span className="text-sm font-semibold text-gray-700">
                              {(obj.confidence * 100).toFixed(1)}% confidence
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </Card>
      )}
    </>
  );
}