import { useState, useRef } from "react";
import { Upload, Camera, AlertTriangle, Flame, Cloud, Loader2, CheckCircle, XCircle } from "lucide-react";
import { LandingPage } from "./features/landing/components/landingPage";
import { Button } from "./components/ui/button";
import { Card } from "./components/ui/card";
import { Badge } from "./components/ui/badge";

type DetectionResult = {
  classification: "fire" | "smoke" | "regular";
  confidence: number;
  localization?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

type ProcessingStage = "classifying" | "localizing" | "complete" | null;

export default function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [processingStage, setProcessingStage] = useState<ProcessingStage>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const mockDetection = (imageData: string): DetectionResult => {
    // Simulate ML model processing time
    const classifications: ("fire" | "smoke" | "regular")[] = ["fire", "smoke", "regular"];
    const randomClass = classifications[Math.floor(Math.random() * classifications.length)];
    
    // If fire or smoke detected, add localization
    const hasDetection = randomClass !== "regular";
    
    return {
      classification: randomClass,
      confidence: 0.75 + Math.random() * 0.2, // 75-95% confidence
      localization: hasDetection ? {
        x: 0.2 + Math.random() * 0.3, // 20-50% from left
        y: 0.15 + Math.random() * 0.3, // 15-45% from top
        width: 0.2 + Math.random() * 0.2, // 20-40% width
        height: 0.2 + Math.random() * 0.2, // 20-40% height
      } : undefined,
    };
  };

  const handleImageSelect = async (file: File) => {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      const imageUrl = e.target?.result as string;
      setSelectedImage(imageUrl);
      setProcessingStage("classifying");
      setResult(null);

      // Stage 1: Classification (API call 1)
      setTimeout(() => {
        const detection = mockDetection(imageUrl);
        
        // If no fire/smoke detected, stop here
        if (detection.classification === "regular") {
          setResult(detection);
          setProcessingStage("complete");
        } else {
          // If fire/smoke detected, move to localization stage
          setResult(detection);
          setProcessingStage("localizing");
          
          // Stage 2: Localization (API call 2)
          setTimeout(() => {
            setProcessingStage("complete");
          }, 1500);
        }
      }, 1500);
    };
    reader.readAsDataURL(file);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleImageSelect(file);
    }
  };

  const getClassificationColor = (classification: string) => {
    switch (classification) {
      case "fire":
        return "bg-red-500";
      case "smoke":
        return "bg-gray-600";
      default:
        return "bg-green-500";
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
    <div className="min-h-screen bg-linear-to-br from-orange-50 via-red-50 to-yellow-50 p-4 pb-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <LandingPage/>
        {/* Upload Section */}
        <Card className="p-8 shadow-2xl border-0 bg-white/80 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-4">
            <Button
              onClick={() => fileInputRef.current?.click()}
              className="w-full sm:w-auto sm:min-w-[300px] h-auto py-8 flex-col gap-3 bg-linear-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white shadow-lg hover:shadow-xl transition-all duration-300 border-0 group"
            >
              <div className="relative">
                <Upload className="w-8 h-8 group-hover:scale-110 transition-transform duration-300" />
              </div>
              <div className="space-y-1">
                <span className="block">Upload Image</span>
                <span className="text-xs opacity-90">From your device</span>
              </div>
            </Button>
            
            <Button
              onClick={() => cameraInputRef.current?.click()}
              className="w-full h-auto py-8 flex-col gap-3 bg-linear-to-br from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white shadow-lg hover:shadow-xl transition-all duration-300 border-0 sm:hidden group"
            >
              <div className="relative">
                <Camera className="w-8 h-8 group-hover:scale-110 transition-transform duration-300" />
              </div>
              <div className="space-y-1">
                <span className="block">Take Photo</span>
                <span className="text-xs opacity-90">Use your camera</span>
              </div>
            </Button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />
          
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleFileUpload}
            className="hidden"
          />
        </Card>

        {/* Results Section */}
        {selectedImage && (
          <Card className="p-8 shadow-2xl border-0 bg-white/80 backdrop-blur-sm overflow-hidden">
            <div className="flex items-center justify-center gap-2 mb-6">
              <div className="h-1 w-12 bg-linear-to-r from-transparent to-orange-500 rounded-full"></div>
              <h2 className="text-center text-gray-800">Analysis Results</h2>
              <div className="h-1 w-12 bg-linear-to-l from-transparent to-orange-500 rounded-full"></div>
            </div>

            {/* Processing Stage Messages */}
            {processingStage === "classifying" && (
              <div className="flex flex-col items-center justify-center py-16 space-y-6 animate-fade-in">
                <div className="relative">
                  <Loader2 className="w-16 h-16 text-orange-600 animate-spin" />
                  <div className="absolute inset-0 w-16 h-16 bg-orange-400 blur-2xl opacity-30 animate-pulse"></div>
                </div>
                <div className="text-center space-y-2">
                  <p className="text-gray-900">Classifying image...</p>
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
                  <p className="text-green-700">{result.classification.charAt(0).toUpperCase() + result.classification.slice(1)} detected!</p>
                  <p className="text-gray-900">Localizing fire sources...</p>
                  <p className="text-sm text-gray-600">Finding the origin location</p>
                  <div className="flex gap-1 justify-center mt-4">
                    <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                </div>
              </div>
            )}

            {/* No Fire Detected Message */}
            {processingStage === "complete" && result && result.classification === "regular" && (
              <div className="flex flex-col items-center justify-center py-16 space-y-6 animate-fade-in">
                <div className="relative">
                  <div className="absolute inset-0 bg-green-400 blur-3xl opacity-20 animate-pulse"></div>
                  <div className="bg-linear-to-br from-green-50 to-emerald-50 p-6 rounded-full relative">
                    <CheckCircle className="w-20 h-20 text-green-600" />
                  </div>
                </div>
                <div className="text-center space-y-3">
                  <p className="text-gray-900">All Clear!</p>
                  <p className="text-sm text-gray-600 max-w-md">No fire or smoke detected in this image. The area appears to be safe.</p>
                </div>
                <div className="space-y-3 w-full mt-8">
                  <div className="flex items-center justify-between p-5 bg-linear-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm border border-gray-200">
                    <div className="flex items-center gap-3">
                      <div className="bg-green-100 p-2 rounded-lg">
                        {getClassificationIcon(result.classification)}
                      </div>
                      <span className="text-gray-700">Classification</span>
                    </div>
                    <Badge className="bg-linear-to-r from-green-500 to-emerald-500 text-white capitalize shadow-md px-4 py-1">
                      Safe
                    </Badge>
                  </div>

                  <div className="flex items-center justify-between p-5 bg-linear-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm border border-gray-200">
                    <div className="flex items-center gap-3">
                      <div className="bg-blue-100 p-2 rounded-lg">
                        <AlertTriangle className="w-4 h-4 text-blue-600" />
                      </div>
                      <span className="text-gray-700">Confidence</span>
                    </div>
                    <span className="text-gray-900 px-3 py-1 bg-white rounded-lg shadow-sm">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            )}

            {/* Image with Localization (only shown when fire/smoke detected and complete) */}
            {processingStage === "complete" && result && result.classification !== "regular" && (
              <div className="space-y-6 animate-fade-in">
                <div className="relative w-full rounded-xl overflow-hidden shadow-2xl bg-gray-100 ring-4 ring-orange-100">
                  <img
                    src={selectedImage}
                    alt="Uploaded forest"
                    className="w-full h-auto"
                  />
                  
                  {/* Localization Box */}
                  {result?.localization && (
                    <div
                      className="absolute border-4 border-red-500 rounded-lg shadow-2xl animate-pulse"
                      style={{
                        left: `${result.localization.x * 100}%`,
                        top: `${result.localization.y * 100}%`,
                        width: `${result.localization.width * 100}%`,
                        height: `${result.localization.height * 100}%`,
                        boxShadow: '0 0 30px rgba(239, 68, 68, 0.6)',
                      }}
                    >
                      <div className="absolute -top-10 left-0 bg-linear-to-r from-red-500 to-orange-500 text-white px-4 py-2 rounded-lg shadow-lg">
                        <Flame className="w-4 h-4 inline mr-1" />
                        Origin Detected
                      </div>
                    </div>
                  )}
                </div>

                {/* Classification Results */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-5 bg-linear-to-br from-red-50 to-orange-50 rounded-xl shadow-md border-2 border-red-200">
                    <div className="flex items-center gap-3">
                      <div className="bg-red-100 p-3 rounded-lg">
                        {getClassificationIcon(result.classification)}
                      </div>
                      <span className="text-gray-800">Threat Detected</span>
                    </div>
                    <Badge className="bg-linear-to-r from-red-500 to-orange-500 text-white capitalize shadow-lg px-4 py-2">
                      {result.classification}
                    </Badge>
                  </div>

                  <div className="flex items-center justify-between p-5 bg-linear-to-br from-gray-50 to-gray-100 rounded-xl shadow-md border border-gray-200">
                    <div className="flex items-center gap-3">
                      <div className="bg-blue-100 p-3 rounded-lg">
                        <AlertTriangle className="w-4 h-4 text-blue-600" />
                      </div>
                      <span className="text-gray-800">Confidence Score</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-linear-to-r from-orange-500 to-red-500 rounded-full transition-all duration-1000"
                          style={{ width: `${result.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-gray-900 px-3 py-1 bg-white rounded-lg shadow-sm min-w-16 text-center">{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  {result.localization && (
                    <div className="p-6 bg-linear-to-br from-orange-50 to-red-50 border-2 border-orange-300 rounded-xl shadow-lg">
                      <div className="flex items-start gap-3">
                        <div className="bg-orange-100 p-2 rounded-lg mt-0.5">
                          <AlertTriangle className="w-6 h-6 text-orange-600" />
                        </div>
                        <div className="flex-1">
                          <p className="text-orange-900 mb-1">⚠️ Origin Location Identified</p>
                          <p className="text-sm text-orange-800">
                            The approximate origin of the {result.classification} has been marked with a red bounding box on the image above. Immediate action recommended.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </Card>
        )}

        {/* Info Section */}
        {!selectedImage && (
          <Card className="p-8 shadow-2xl border-0 bg-linear-to-br from-blue-50 to-indigo-50 backdrop-blur-sm">
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <div className="h-1 w-8 bg-linear-to-r from-blue-500 to-indigo-500 rounded-full"></div>
                <h3 className="text-transparent bg-clip-text bg-linear-to-r from-blue-600 to-indigo-600">How It Works</h3>
              </div>
              <div className="grid gap-4 mt-6">
                <div className="flex items-start gap-4 p-4 bg-white/60 rounded-xl shadow-sm border border-blue-100">
                  <div className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center shrink-0 shadow-md">1</div>
                  <div>
                    <p className="text-blue-900">Upload or Capture</p>
                    <p className="text-sm text-blue-700">Select a forest image from your device or take a photo with your camera</p>
                  </div>
                </div>
                <div className="flex items-start gap-4 p-4 bg-white/60 rounded-xl shadow-sm border border-blue-100">
                  <div className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center shrink-0 shadow-md">2</div>
                  <div>
                    <p className="text-blue-900">AI Classification</p>
                    <p className="text-sm text-blue-700">Our deep learning model analyzes the image for fire, smoke, or safe conditions</p>
                  </div>
                </div>
                <div className="flex items-start gap-4 p-4 bg-white/60 rounded-xl shadow-sm border border-blue-100">
                  <div className="bg-purple-500 text-white rounded-full w-8 h-8 flex items-center justify-center shrink-0 shadow-md">3</div>
                  <div>
                    <p className="text-blue-900">Threat Localization</p>
                    <p className="text-sm text-blue-700">If a threat is detected, the system pinpoints the approximate origin location</p>
                  </div>
                </div>
                <div className="flex items-start gap-4 p-4 bg-white/60 rounded-xl shadow-sm border border-blue-100">
                  <div className="bg-pink-500 text-white rounded-full w-8 h-8 flex items-center justify-center shrink-0 shadow-md">4</div>
                  <div>
                    <p className="text-blue-900">Rapid Response</p>
                    <p className="text-sm text-blue-700">Results enable faster emergency response and help minimize environmental damage</p>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}