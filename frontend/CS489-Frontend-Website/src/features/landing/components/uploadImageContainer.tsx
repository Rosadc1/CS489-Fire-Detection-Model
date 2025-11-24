import { useRef } from "react";
import { Upload, Camera } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface UploadImageContainerProps {
  onImageSelect: (file: File) => void;
}

export function UploadImageContainer({ onImageSelect }: UploadImageContainerProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onImageSelect(file);
    }
  };

  return (
    <Card className="p-8 shadow-2xl border-0 bg-white/80 backdrop-blur-sm">
      <div className="flex flex-col items-center gap-4">
        <Button
          onClick={() => fileInputRef.current?.click()}
          className="w-full sm:w-auto sm:min-w-[300px] h-auto py-8 flex-col gap-3 bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white shadow-lg hover:shadow-xl transition-all duration-300 border-0 group"
        >
          <div className="relative">
            <Upload className="w-8 h-8 group-hover:scale-110 transition-transform duration-300" />
          </div>
          <div className="space-y-1">
            <span className="block text-lg font-semibold">Upload Image</span>
            <span className="text-xs opacity-90">From your device</span>
          </div>
        </Button>
        
        <Button
          onClick={() => cameraInputRef.current?.click()}
          className="w-full h-auto py-8 flex-col gap-3 bg-gradient-to-br from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white shadow-lg hover:shadow-xl transition-all duration-300 border-0 sm:hidden group"
        >
          <div className="relative">
            <Camera className="w-8 h-8 group-hover:scale-110 transition-transform duration-300" />
          </div>
          <div className="space-y-1">
            <span className="block text-lg font-semibold">Take Photo</span>
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
  );
}