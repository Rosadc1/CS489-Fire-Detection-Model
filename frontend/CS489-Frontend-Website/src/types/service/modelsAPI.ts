
export type predictModelRequest = {
    image: File
}

export type predictModelResponse = {
    predicted_class:predictedClassType,
    probability_fire:number,
    probability_no_fire:number
} | errorResponse;

export type detectModelRequest = {
    image: File
} 

export type detectModelResponse = {
    image:string
    predicted_boxes:predicted_box[]
} | errorResponse

export type errorResponse = {
    detail: string
}

export type predictedClassType = "fire" | "no_fire";

export type predicted_box = {
    name:boxName,
    class:boxClass,
    confidence:number,
    box:box
}

export type box = { 
    x1:number,
    y1:number,
    x2:number,
    y2:number
}

type boxName = "fire" | "smoke";
type boxClass = 0 | 1;


export type getResponse = { 
    message: string
}

export type getRequest = {
    
}