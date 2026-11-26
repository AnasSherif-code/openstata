"""Built-in command registration for OpenStata."""

from openstata.commands.data_cmds import (
    AppendCmd,
    CollapseCmd,
    DecodeCmd,
    DestringCmd,
    DropCmd,
    DuplicatesCmd,
    EgenCmd,
    EncodeCmd,
    GenerateCmd,
    GsortCmd,
    KeepCmd,
    LabelCmd,
    MergeCmd,
    RenameCmd,
    ReplaceCmd,
    ReshapeCmd,
    SortCmd,
    TostringCmd,
)
from openstata.commands.io_cmds import (
    CodebookCmd,
    CountCmd,
    DescribeCmd,
    ExportCmd,
    ImportCmd,
    ListCmd,
    SaveCmd,
    UseCmd,
)
from openstata.commands.model_cmds import (
    BinaryChoiceCmd,
    PredictCmd,
    RegressCmd,
    XtregCmd,
    XtsetCmd,
)
from openstata.commands.stats_cmds import (
    CorrelateCmd,
    SummarizeCmd,
    TabulateCmd,
    TtestCmd,
)
from openstata.causal.didregress import DidregressCmd
from openstata.causal.gencohort import GencohortCmd

from openstata.commands.system_cmds import (
    AdopathCmd,
    ArgsCmd,
    AssertCmd,
    CdCmd,
    ClearCmd,
    DisplayCmd,
    DoCmd,
    EreturnCmd,
    ExitCmd,
    GlobalCmd,
    HelpCmd,
    LocalCmd,
    LogCmd,
    MarksampleCmd,
    PkgCmd,
    PwdCmd,
    ReturnCmd,
    RunCmd,
    ScalarCmd,
    SetCmd,
    SyntaxCmd,
    TempfileCmd,
    TempnameCmd,
    TempvarCmd,
    VersionCmd,
    WhichCmd,
)
from openstata.plugins.registry import CommandRegistry


def register_builtin_commands(registry: CommandRegistry) -> None:
    """Register all built-in OpenStata commands."""
    commands = [
        # System & scripting
        ClearCmd(),
        VersionCmd(),
        SetCmd(),
        CdCmd(),
        PwdCmd(),
        DisplayCmd(),
        LocalCmd(),
        GlobalCmd(),
        ScalarCmd(),
        TempvarCmd(),
        TempnameCmd(),
        TempfileCmd(),
        ReturnCmd(),
        EreturnCmd(),
        ArgsCmd(),
        SyntaxCmd(),
        MarksampleCmd(),
        DoCmd(),
        RunCmd(),
        LogCmd(),
        ExitCmd(),
        AssertCmd(),
        WhichCmd(),
        HelpCmd(),
        AdopathCmd(),

        # Causal inference
        DidregressCmd(),
        GencohortCmd(),
        # I/O & inspection
        UseCmd(),
        SaveCmd(),
        ImportCmd(),
        ExportCmd(),
        DescribeCmd(),
        ListCmd(),
        CountCmd(),
        CodebookCmd(),
        # Data manipulation
        GenerateCmd(),
        ReplaceCmd(),
        EgenCmd(),
        DropCmd(),
        KeepCmd(),
        RenameCmd(),
        SortCmd(),
        GsortCmd(),
        LabelCmd(),
        EncodeCmd(),
        DecodeCmd(),
        DestringCmd(),
        TostringCmd(),
        DuplicatesCmd(),
        MergeCmd(),
        AppendCmd(),
        CollapseCmd(),
        ReshapeCmd(),
        # Statistics & econometrics
        SummarizeCmd(),
        TabulateCmd(),
        CorrelateCmd(),
        TtestCmd(),
        RegressCmd(),
        BinaryChoiceCmd("logit"),
        BinaryChoiceCmd("probit"),
        XtsetCmd(),
        XtregCmd(),
        PredictCmd(),
    ]
    for cmd in commands:
        registry.register(cmd)
