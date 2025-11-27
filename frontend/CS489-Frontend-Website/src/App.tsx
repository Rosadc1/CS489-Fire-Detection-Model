import { useState } from "react";
import { WaveBackground } from "./features/landing/components/WaveBackground";
import { LandingPage } from "./features/landing/components/landingPage";
import { UploadImageContainer } from "./features/landing/components/uploadImageContainer";
import { ResultsContainer } from "./features/landing/components/resultsContainer";
import { InfoCard } from "./features/landing/components/infoCard";
import { usePredictMutation, useDetectMutation } from "./service/modelsAPI";

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

export default function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [processingStage, setProcessingStage] = useState<ProcessingStage>(null);
  const [error, setError] = useState<string>("");

  // RTK Query hooks
  const [predict] = usePredictMutation();
  const [detect] = useDetectMutation();

  const handleImageSelect = async (file: File) => {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      const imageUrl = e.target?.result as string;
      setSelectedImage(imageUrl);
      setProcessingStage("classifying");
      setResult(null);
      setError("");

      try {
        // Classification Stage
        const classifyResult = await predict({ image: file }).unwrap();
        
        if ('detail' in classifyResult) {
          throw new Error(classifyResult.detail);
        }
      
        if (classifyResult.predicted_class === "no_fire") {
          setResult({
            classification: "no_fire",
            confidence: classifyResult.probability_no_fire,
          });
          setProcessingStage("complete");
        } else {
          // Fire detected
          setResult({
            classification: "fire",
            confidence: classifyResult.probability_fire,
          });
          setProcessingStage("localizing");
          
          // Object Detection Stage
          const detectResult = await detect({ image: file }).unwrap();
          
          if ('detail' in detectResult) {
            throw new Error(detectResult.detail);
          }

          setResult({
            classification: "fire",
            confidence: classifyResult.probability_fire,
            localization: {
              image: detectResult.image,
              detectedObjects: detectResult.predicted_boxes,
            },
          });
          setProcessingStage("complete");
        }
      } catch (err: any) {
        console.error('API Error:', err);
        
        let errorMessage = "Analysis failed";
        
        if (err.data) {
          if (err.data.detail) {
            if (typeof err.data.detail === 'string') {
              errorMessage = err.data.detail;
            } else if (Array.isArray(err.data.detail)) {
              errorMessage = err.data.detail.map((e: any) => e.msg).join(', ');
            } else {
              errorMessage = JSON.stringify(err.data.detail);
            }
          }
        } else if (err.message) {
          errorMessage = err.message;
        }
        
        setError(errorMessage);
        setProcessingStage(null);
      }
    };
    reader.readAsDataURL(file);
  };

  return (
    <>
      <WaveBackground />
      
      <div className="relative min-h-screen p-4 pb-8">
        <div className="max-w-4xl mx-auto space-y-8">
          <LandingPage />
          <UploadImageContainer onImageSelect={handleImageSelect} />
          
          {(selectedImage || error) && (
            <ResultsContainer
              selectedImage={selectedImage!}
              result={result}
              processingStage={processingStage}
              error={error}
            />
          )}
          
         {!selectedImage && <InfoCard />}
        </div>
      </div>
    </>
  );
}