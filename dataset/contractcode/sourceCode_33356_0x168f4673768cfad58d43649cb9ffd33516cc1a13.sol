/**
 *Submitted for verification at Etherscan.io on 2018-08-21
*/

pragma solidity 0.4.24;
// produced by the Solididy File Flattener (c) David Appleton 2018
// contact : [email protected]
// released under Apache 2.0 licence
contract Migrations {
    address public owner;
    uint public last_completed_migration;

    modifier restricted() {
        if (msg.sender == owner) _;
    }

    constructor() public {
        owner = msg.sender;
    }

    function setCompleted(uint completed) public restricted {
        last_completed_migration = completed;
    }

    function upgrade(address new_address) public restricted {
        Migrations upgraded = Migrations(new_address);
        upgraded.setCompleted(last_completed_migration);
    }
}