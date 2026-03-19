let
    Source = Excel.Workbook(File.Contents(Folder.Path & "outputs/teams_power_full.xlsx"), null, true),
    Data = Source{[Item="data", Kind="Sheet"]}[Data],
    Promoted = Table.PromoteHeaders(Data, [PromoteAllScalars=true]),
    Types = Table.TransformColumnTypes(Promoted,{
        {"Rank", Int64.Type},
        {"team_id", type text},
        {"team", type text},
        {"AdjOE", type number},
        {"AdjDE", type number},
        {"AdjEM", type number},
        {"Pace", type number},
        {"Games", Int64.Type},
        {"SOS_Power", type number}
    })
in
    Types
