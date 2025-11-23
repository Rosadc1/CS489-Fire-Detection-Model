import { Flame } from "lucide-react";
import { LandingPageTitle } from "../constants/landingPage";
import { Button } from "@/components/ui/button";
import { useLazyRootQuery } from "@/service/modelsAPI";

export function LandingPage() {
    const testFile = new File([], "tempName");
    const [getRequestCall, {isLoading}] = useLazyRootQuery();
    return(
        <div className="text-center space-y-4 pt-8">
            <div className="flex items-center justify-center gap-3 mb-2">
                <div className="relative">
                <Flame className="w-12 h-12 text-orange-600 drop-shadow-lg" />
                <div className="absolute inset-0 w-12 h-12 bg-orange-400 blur-xl opacity-50 animate-pulse"></div>
                </div>
            </div>
            <h1 className="text-transparent bg-clip-text bg-linear-to-r from-orange-600 to-red-600">
                Forest Fire Detection System
            </h1>
            <p className="text-gray-700 max-w-2xl mx-auto text-lg">
                {LandingPageTitle}
            </p>
            <Button onClick={() => getRequestCall({})}>Click me!</Button>
            {isLoading && <div>loading...</div>}
        </div>
    )
}