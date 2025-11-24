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

interface ResultsContainerProps {
  selectedImage: string;
  result: DetectionResult | null;
  processingStage: ProcessingStage;
  error: string;
}

export function ResultsContainer({
  selectedImage,
  result,
  processingStage,
  error,
}: ResultsContainerProps) {
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
          <div className="flex items-center justify-center gap-2 mb-6">
            <div className="h-1 w-12 bg-gradient-to-r from-transparent to-orange-500 rounded-full"></div>
            <h2 className="text-2xl font-bold text-center text-gray-800">Analysis Results</h2>
            <div className="h-1 w-12 bg-gradient-to-l from-transparent to-orange-500 rounded-full"></div>
          </div>

          {processingStage === "classifying" && (
            <div className="flex flex-col items-center justify-center py-16 space-y-6 animate-fade-in">
              <div className="relative">
                <Loader2 className="w-16 h-16 text-orange-600 animate-spin" />
                <div className="absolute inset-0 w-16 h-16 bg-orange-400 blur-2xl opacity-30 animate-pulse"></div>
              </div>
              <div className="text-center space-y-2">
                <p className="text-xl font-semibold text-gray-900">Classifying image...</p>
                <p className="text-sm text-gray-600">Analyzing for fire and smoke detection</p>
                <div className="flex gap-1 justify-center mt-4">
                  <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}

          {processingStage === "localizing" && result && (
            <div className="flex flex-col items-center justify-center py-16 space-y-6 animate-fade-in">
              <div className="relative">
                <div className="absolute inset-0 bg-green-400 blur-2xl opacity-30 animate-pulse"></div>
                <CheckCircle className="w-16 h-16 text-green-600 relative" />
                <div className="absolute -bottom-2 -right-2 bg-white rounded-full p-1 shadow-lg">
                  <Loader2 className="w-8 h-8 text-orange-600 animate-spin" />
                </div>
              </div>
              <div className="text-center space-y-2">
                <p className="text-xl font-semibold text-green-700">Fire detected!</p>
                <p className="text-lg text-gray-900">Localizing fire sources...</p>
                <p className="text-sm text-gray-600">Finding the origin location</p>
                <div className="flex gap-1 justify-center mt-4">
                  <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}

          {processingStage === "complete" && result && result.classification === "no_fire" && (
            <div className="flex flex-col items-center justify-center py-16 space-y-6 animate-fade-in">
              <div className="relative">
                <div className="absolute inset-0 bg-green-400 blur-3xl opacity-20 animate-pulse"></div>
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-full relative">
                  <CheckCircle className="w-20 h-20 text-green-600" />
                </div>
              </div>
              <div className="text-center space-y-3">
                <p className="text-2xl font-bold text-gray-900">All Clear!</p>
                <p className="text-sm text-gray-600 max-w-md">No fire or smoke detected in this image. The area appears to be safe.</p>
              </div>
              <div className="space-y-3 w-full mt-8">
                <div className="flex items-center justify-between p-5 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm border border-gray-200">
                  <div className="flex items-center gap-3">
                    <div className="bg-green-100 p-2 rounded-lg">
                      {getClassificationIcon(result.classification)}
                    </div>
                    <span className="font-medium text-gray-700">Classification</span>
                  </div>
                  <Badge className="bg-gradient-to-r from-green-500 to-emerald-500 text-white capitalize shadow-md px-4 py-1">
                    Safe
                  </Badge>
                </div>

                <div className="flex items-center justify-between p-5 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm border border-gray-200">
                  <div className="flex items-center gap-3">
                    <div className="bg-blue-100 p-2 rounded-lg">
                      <AlertTriangle className="w-4 h-4 text-blue-600" />
                    </div>
                    <span className="font-medium text-gray-700">Confidence</span>
                  </div>
                  <span className="font-bold text-gray-900 px-3 py-1 bg-white rounded-lg shadow-sm">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )}

          {processingStage === "complete" && result && result.classification === "fire" && result.localization && (
            <div className="space-y-6 animate-fade-in">
              <div className="relative w-full rounded-xl overflow-hidden shadow-2xl bg-gray-100 ring-4 ring-orange-100">
                <img
                  src={`data:image/jpeg;base64,${result.localization.image}`}
                  alt="Fire detection result"
                  className="w-full h-auto"
                />
                
                {result.localization.detectedObjects.map((obj, idx) => {
                  const boxWidth = obj.box.x2 - obj.box.x1;
                  const boxHeight = obj.box.y2 - obj.box.y1;
                  
                  return (
                    <div
                      key={idx}
                      className="absolute border-4 rounded-lg shadow-2xl animate-pulse"
                      style={{
                        left: `${obj.box.x1}px`,
                        top: `${obj.box.y1}px`,
                        width: `${boxWidth}px`,
                        height: `${boxHeight}px`,
                        borderColor: obj.name === 'fire' ? '#ef4444' : '#6b7280',
                        boxShadow: obj.name === 'fire' 
                          ? '0 0 30px rgba(239, 68, 68, 0.6)' 
                          : '0 0 30px rgba(107, 114, 128, 0.6)',
                      }}
                    >
                      <div 
                        className="absolute -top-10 left-0 text-white px-4 py-2 rounded-lg shadow-lg text-sm font-semibold"
                        style={{
                          background: obj.name === 'fire' 
                            ? 'linear-gradient(to right, #ef4444, #f97316)' 
                            : 'linear-gradient(to right, #6b7280, #4b5563)'
                        }}
                      >
                        {obj.name === 'fire' ? (
                          <><Flame className="w-4 h-4 inline mr-1" /> Fire Detected</>
                        ) : (
                          <><Cloud className="w-4 h-4 inline mr-1" /> Smoke Detected</>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-5 bg-gradient-to-br from-red-50 to-orange-50 rounded-xl shadow-md border-2 border-red-200">
                  <div className="flex items-center gap-3">
                    <div className="bg-red-100 p-3 rounded-lg">
                      {getClassificationIcon(result.classification)}
                    </div>
                    <span className="font-semibold text-gray-800">Threat Detected</span>
                  </div>
                  <Badge className="bg-gradient-to-r from-red-500 to-orange-500 text-white capitalize shadow-lg px-4 py-2">
                    Fire
                  </Badge>
                </div>

                <div className="flex items-center justify-between p-5 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-md border border-gray-200">
                  <div className="flex items-center gap-3">
                    <div className="bg-blue-100 p-3 rounded-lg">
                      <AlertTriangle className="w-4 h-4 text-blue-600" />
                    </div>
                    <span className="font-semibold text-gray-800">Confidence Score</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-orange-500 to-red-500 rounded-full transition-all duration-1000"
                        style={{ width: `${result.confidence * 100}%` }}
                      ></div>
                    </div>
                    <span className="font-bold text-gray-900 px-3 py-1 bg-white rounded-lg shadow-sm min-w-[4rem] text-center">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {result.localization.detectedObjects.length > 0 && (
                  <div className="p-6 bg-gradient-to-br from-orange-50 to-red-50 border-2 border-orange-300 rounded-xl shadow-lg">
                    <div className="flex items-start gap-3 mb-4">
                      <div className="bg-orange-100 p-2 rounded-lg mt-0.5">
                        <AlertTriangle className="w-6 h-6 text-orange-600" />
                      </div>
                      <div className="flex-1">
                        <p className="font-bold text-orange-900 mb-1">
                          ⚠️ {result.localization.detectedObjects.length} Object(s) Detected
                        </p>
                        <p className="text-sm text-orange-800">
                          The detection boxes have been marked on the image above. Immediate action recommended.
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
        </Card>
      )}
    </>
  );
}