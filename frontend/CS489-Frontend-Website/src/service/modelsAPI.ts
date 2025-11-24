import type { detectModelRequest, detectModelResponse, getRequest, getResponse, predictModelRequest, predictModelResponse } from "@/types/service/modelsAPI";
import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";

export const modelsAPI = createApi({
    reducerPath: "modelsAPI",
    tagTypes: [],
    baseQuery: fetchBaseQuery({
        baseUrl: "http://cs-489-load-balancing-545464512.us-west-1.elb.amazonaws.com"
    }),
    endpoints: (build) => ({
        predict: build.mutation<predictModelResponse, predictModelRequest>({
            query: (request) => ({
                url: `/predict`,
                method: "POST",
                body: request
            })
        }),
        detect: build.mutation<detectModelResponse, detectModelRequest>({
            query: (request) => ({
                url: `/detect`,
                method: "POST",
                body: request
            })
        }),
        root: build.query<getResponse, getRequest>({
            query: () =>({
                url: `/`,
                method: "GET",
            })
        })
    })
});

export const {  useDetectMutation, usePredictMutation, useLazyRootQuery } = modelsAPI